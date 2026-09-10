import { strapi } from "@strapi/client";
import { GoogleGenAI } from "@google/genai";
import {
  SecretsManagerClient,
  GetSecretValueCommand,
} from "@aws-sdk/client-secrets-manager";
import fs from "fs";
import path from "path";
import qs from "qs";

const PROJECT_ID = process.env.VERTEX_AI_PROJECT_ID || "expath-app";
const LOCATION = process.env.VERTEX_AI_LOCATION || "australia-southeast1";
// gemini-embedding-001 (multilingual — the old text-embedding-004 was
// English-only and buried zh/ar queries) truncated to the platform's 768
// dims via Matryoshka. Queries embed with the SAME model+dims (Strapi
// controllers / facade embedKeyword) — mixed-model vectors are noise.
const MODEL = "gemini-embedding-001";
const EMBED_CONFIG = { taskType: "RETRIEVAL_DOCUMENT", outputDimensionality: 768 };
const BATCH_SIZE = 50;

// Truncated Matryoshka outputs are not unit-length; normalize so pgvector's
// L2 `<->` ordering is equivalent to cosine similarity.
function normalizeEmbedding(values) {
  const norm = Math.sqrt(values.reduce((sum, v) => sum + v * v, 0));
  return norm > 0 ? values.map((v) => v / norm) : values;
}

let strapiClient = null;
let aiClient = null;
let medusaAuthHeader = null;
let initialized = false;

const secretsClient = new SecretsManagerClient({
  region: process.env.AWS_REGION || "ap-southeast-2",
});

async function initClients() {
  if (initialized) return;
  console.log("Initializing clients and fetching secrets...");

  // 1. Fetch Strapi Token
  const strapiSecretCmd = new GetSecretValueCommand({
    SecretId: process.env.STRAPI_API_TOKEN_SECRET_ARN,
  });
  const strapiSecretRes = await secretsClient.send(strapiSecretCmd);
  const strapiApiToken = strapiSecretRes.SecretString;

  strapiClient = strapi({
    baseURL: process.env.STRAPI_API_URL || "http://localhost:1337/api",
    auth: strapiApiToken,
  });

  // 2. Fetch Vertex Credentials
  const vertexSecretCmd = new GetSecretValueCommand({
    SecretId: process.env.VERTEX_CREDENTIALS_SECRET_ARN,
  });
  const vertexSecretRes = await secretsClient.send(vertexSecretCmd);
  const aiServiceAccountB64 = vertexSecretRes.SecretString;

  // Write base64 credentials to a temp file
  const credsJson = Buffer.from(aiServiceAccountB64, "base64").toString("utf8");
  const tempCredsPath = path.join("/tmp", "google-credentials.json");
  fs.writeFileSync(tempCredsPath, credsJson, { mode: 0o600 });

  // Set environment variable for GoogleGenAI to find
  process.env.GOOGLE_APPLICATION_CREDENTIALS = tempCredsPath;

  aiClient = new GoogleGenAI({
    vertexai: true,
    project: PROJECT_ID,
    location: LOCATION,
  });

  // 3. Medusa admin key (shop products + shops). Optional: without the env
  // vars the Medusa pass is skipped, so this lambda still deploys against
  // environments that predate the shop.
  if (process.env.MEDUSA_API_URL && process.env.MEDUSA_KEYS_SECRET_ARN) {
    const medusaSecretCmd = new GetSecretValueCommand({
      SecretId: process.env.MEDUSA_KEYS_SECRET_ARN,
    });
    const medusaSecretRes = await secretsClient.send(medusaSecretCmd);
    const adminKey = JSON.parse(
      medusaSecretRes.SecretString,
    ).MEDUSA_ADMIN_API_KEY;
    if (adminKey) {
      medusaAuthHeader = `Basic ${Buffer.from(`${adminKey}:`).toString("base64")}`;
    }
  }

  initialized = true;
  console.log("Clients initialized successfully.");
}

/**
 * Shop content (products + shops) lives in MEDUSA, not Strapi, and its
 * endpoint returns READY-BUILT text — so unlike the Strapi models there is
 * no mapper here, just embed-and-post:
 *   GET  {MEDUSA_API_URL}/admin/expath-embeddings?limit=50
 *          → { items: [{ type: 'product'|'seller', id, text }] }
 *   POST {MEDUSA_API_URL}/admin/expath-embeddings   { type, id, embedding }
 */
async function processMedusaShop() {
  if (!medusaAuthHeader) {
    console.log("Medusa not configured — skipping shop embeddings.");
    return;
  }
  const baseUrl = process.env.MEDUSA_API_URL;
  const listRes = await fetch(
    `${baseUrl}/admin/expath-embeddings?limit=${BATCH_SIZE}`,
    {
      headers: { authorization: medusaAuthHeader },
    },
  );
  if (!listRes.ok) {
    console.error(
      `Medusa unembedded fetch failed: ${listRes.status} ${await listRes.text()}`,
    );
    return;
  }
  const { items } = await listRes.json();
  if (!items?.length) {
    console.log("No unembedded shop items found.");
    return;
  }
  console.log(`Found ${items.length} unembedded shop item(s).`);

  let successCount = 0;
  for (const item of items) {
    try {
      const response = await aiClient.models.embedContent({
        model: MODEL,
        contents: item.text,
        config: EMBED_CONFIG,
      });
      const embeddingValues = normalizeEmbedding(response.embeddings[0].values);
      const saveRes = await fetch(`${baseUrl}/admin/expath-embeddings`, {
        method: "POST",
        headers: {
          authorization: medusaAuthHeader,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          type: item.type,
          id: item.id,
          embedding: embeddingValues,
        }),
      });
      if (!saveRes.ok)
        throw new Error(
          `save failed: ${saveRes.status} ${await saveRes.text()}`,
        );
      successCount++;
    } catch (error) {
      console.error(`Failed to embed shop ${item.type} ${item.id}:`, error);
    }
  }
  console.log(
    `🎉 Finished shop batch: embedded ${successCount}/${items.length}`,
  );
}

/**
 * Hash contract (videos only): when Strapi hands us `{documentId, text, hash}`
 * we post the hash back and Strapi applies the vector with
 * `UPDATE … WHERE embedding_desired_hash = hash`. A 409 means the row's
 * desired text changed while we were embedding — the next poll picks it up
 * again with the new text, so it is "stale", not a failure. Without a hash
 * (jobs/properties/providers/profiles, or a Strapi that predates the
 * contract) the write is unconditional, as before.
 */
async function updateStrapiEmbedding(modelName, documentId, embeddingValues, hash) {
  const body = { embedding: embeddingValues };
  if (hash) body.hash = hash;
  const response = await strapiClient.fetch(
    `/${modelName}/${documentId}/embedding`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    },
  );

  if (response.status === 409 && hash) return "stale";

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }
  return "ok";
}

/**
 * @param {object} [options]
 * @param {boolean} [options.serverText] — the endpoint may return rows shaped
 *   `{documentId, text, hash}` (text built server-side). Rows carrying `text`
 *   are embedded verbatim and acknowledged with their `hash`; rows without
 *   it (older Strapi) fall back to `mapFn` and an unconditional write.
 */
async function processModel(modelName, populateConfig, mapFn, options = {}) {
  console.log(`\n🔍 Fetching ${modelName} without embeddings...`);

  const queryParams = qs.stringify(
    {
      populate: populateConfig,
      pagination: { page: 1, pageSize: BATCH_SIZE },
      limit: BATCH_SIZE,
    },
    { encodeValuesOnly: true },
  );

  const response = await strapiClient.fetch(
    `/${modelName}/unembedded?${queryParams}`,
  );

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(
      `HTTP ${response.status} Failed to fetch unembedded ${modelName}: ${errText}`,
    );
  }

  const payload = await response.json();
  const data = Array.isArray(payload) ? payload : payload?.data;

  if (!data || data.length === 0) {
    console.log(`✅ No unembedded ${modelName} found.`);
    return;
  }

  console.log(`📄 Found ${data.length} unembedded ${modelName}, processing...`);

  let successCount = 0;
  let staleCount = 0;
  for (const item of data) {
    const { id, createdAt, updatedAt, publishedAt, embedding, ...rest } = item;
    const serverText =
      options.serverText && typeof item.text === "string" ? item.text : null;
    const hash = serverText && typeof item.hash === "string" ? item.hash : undefined;
    const textToEmbed = serverText ?? JSON.stringify(mapFn(rest));

    try {
      const response = await aiClient.models.embedContent({
        model: MODEL,
        contents: textToEmbed,
        config: EMBED_CONFIG,
      });

      if (response.embeddings && response.embeddings.length > 0) {
        const embeddingValues = normalizeEmbedding(response.embeddings[0].values);
        const outcome = await updateStrapiEmbedding(
          modelName,
          item.documentId,
          embeddingValues,
          hash,
        );
        if (outcome === "stale") {
          staleCount++;
          console.log(
            `   ↻ Stale hash for ${item.documentId} (text changed mid-flight); will retry next run`,
          );
        } else {
          console.log(`   ✅ Embedded & updated: ${item.documentId}`);
          successCount++;
        }
      } else {
        console.log(`   ❌ No embedding returned for ${item.documentId}`);
      }
    } catch (err) {
      console.error(
        `   ❌ Failed to process ${item.documentId}:`,
        err.message || err,
      );
    }
  }

  console.log(
    `🎉 Finished batch for ${modelName}: embedded ${successCount}/${data.length}${staleCount ? ` (${staleCount} stale)` : ""}`,
  );
}

export const handler = async (event, context) => {
  console.log("Starting Embedding Job...");
  await initClients();

  // 1. Process Jobs
  await processModel(
    "jobs",
    ["author", "video", "location", "job_categories"],
    (item) => ({
      ...item,
      location: item.location
        ? {
            documentId: item.location.documentId,
            name: item.location.name,
            city: item.location.city,
            country: item.location.country,
          }
        : null,
      author: item.author
        ? {
            documentId: item.author.documentId,
            nickname: item.author.nickname,
            username: item.author.username,
          }
        : null,
      video: item.video
        ? {
            documentId: item.video.documentId,
            description: item.video.description,
          }
        : null,
      job_categories: item.job_categories
        ? item.job_categories.map((c) => ({
            documentId: c.documentId,
            name: c.name,
          }))
        : null,
    }),
  );

  // 2. Process Properties
  await processModel(
    "properties",
    ["author", "video_tour", "location", "features"],
    (item) => ({
      ...item,
      location: item.location
        ? {
            documentId: item.location.documentId,
            name: item.location.name,
            city: item.location.city,
            country: item.location.country,
          }
        : null,
      author: item.author
        ? {
            documentId: item.author.documentId,
            nickname: item.author.nickname,
            username: item.author.username,
          }
        : null,
      video_tour: item.video_tour
        ? {
            documentId: item.video_tour.documentId,
            description: item.video_tour.description,
          }
        : null,
      features: item.features
        ? item.features.map((f) => ({
            documentId: f.documentId,
            name: f.name,
            type: f.type,
          }))
        : null,
    }),
  );

  // 3. Process Providers
  await processModel(
    "providers",
    {
      profile: { populate: ["locations"] },
      service_deliveries: true,
      payment_methods: true,
      age_groups: true,
      attendant_genders: true,
      specializations: true,
      languages: true,
      service_types: true,
      service_areas: true,
    },
    (item) => ({
      ...item,
      profile: item.profile
        ? {
            documentId: item.profile.documentId,
            nickname: item.profile.nickname,
            description: item.profile.description,
            locations: item.profile.locations
              ? item.profile.locations.map((loc) => ({
                  documentId: loc.documentId,
                  name: loc.name,
                  city: loc.city,
                  country: loc.country,
                }))
              : null,
          }
        : null,
      service_deliveries: item.service_deliveries
        ? item.service_deliveries.map((x) => ({ name: x.name }))
        : null,
      payment_methods: item.payment_methods
        ? item.payment_methods.map((x) => ({ name: x.name }))
        : null,
      age_groups: item.age_groups
        ? item.age_groups.map((x) => ({ name: x.name }))
        : null,
      attendant_genders: item.attendant_genders
        ? item.attendant_genders.map((x) => ({ name: x.name }))
        : null,
      specializations: item.specializations
        ? item.specializations.map((x) => ({ name: x.name }))
        : null,
      languages: item.languages
        ? item.languages.map((x) => ({ name: x.name }))
        : null,
      service_types: item.service_types
        ? item.service_types.map((x) => ({ name: x.name }))
        : null,
      service_areas: item.service_areas
        ? item.service_areas.map((x) => ({
            name: x.name,
            city: x.city,
            country: x.country,
          }))
        : null,
    }),
  );

  // 4. Process Videos. Strapi now builds the text server-side
  // (description + hashtags + transcript_text) and returns
  // {documentId, text, hash}; the mapper below is only the fallback for a
  // Strapi that predates that contract (rows without `text`).
  await processModel("videos", ["author", "location"], (item) => ({
    description: item.description,
    visibility: item.visibility,
    latitude: item.latitude,
    longitude: item.longitude,
    author: item.author
      ? {
          documentId: item.author.documentId,
          username: item.author.username,
          nickname: item.author.nickname,
        }
      : null,
    location: item.location
      ? {
          documentId: item.location.documentId,
          name: item.location.name,
          city: item.location.city,
          country: item.location.country,
        }
      : null,
  }), { serverText: true });

  // 5. Process Profiles - powers personal profile search (searchPersonal). Unlike the
  // provider embedding above (which folds in the whole business profile), this is just
  // username + nickname, since that's all personal profile search ranks/matches on.
  await processModel("profiles", undefined, (item) => ({
    username: item.username,
    nickname: item.nickname,
  }));

  // 6. Process shop products + shops (Medusa; skipped when not configured)
  await processMedusaShop();

  console.log("Finished Embedding Job successfully.");
  return { statusCode: 200, body: "Success" };
};
