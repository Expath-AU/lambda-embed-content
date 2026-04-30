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
const MODEL = "text-embedding-004";
const BATCH_SIZE = 50;

let strapiClient = null;
let aiClient = null;
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

  initialized = true;
  console.log("Clients initialized successfully.");
}

async function updateStrapiEmbedding(modelName, documentId, embeddingValues) {
  const response = await strapiClient.fetch(`/${modelName}/${documentId}/embedding`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ embedding: embeddingValues }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }
}

async function processModel(modelName, populateConfig, mapFn) {
  console.log(`\n🔍 Fetching ${modelName} without embeddings...`);

  const queryParams = qs.stringify(
    {
      populate: populateConfig,
      pagination: { page: 1, pageSize: BATCH_SIZE },
    },
    { encodeValuesOnly: true }
  );

  const response = await strapiClient.fetch(`/${modelName}/unembedded?${queryParams}`);

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`HTTP ${response.status} Failed to fetch unembedded ${modelName}: ${errText}`);
  }

  const { data } = await response.json();

  if (!data || data.length === 0) {
    console.log(`✅ No unembedded ${modelName} found.`);
    return;
  }

  console.log(`📄 Found ${data.length} unembedded ${modelName}, processing...`);

  let successCount = 0;
  for (const item of data) {
    const { id, createdAt, updatedAt, publishedAt, embedding, ...rest } = item;
    const payloadToEmbed = mapFn(rest);
    const textToEmbed = JSON.stringify(payloadToEmbed);

    try {
      const response = await aiClient.models.embedContent({
        model: MODEL,
        contents: textToEmbed,
      });

      if (response.embeddings && response.embeddings.length > 0) {
        const embeddingValues = response.embeddings[0].values;
        await updateStrapiEmbedding(
          modelName,
          item.documentId,
          embeddingValues,
        );
        console.log(`   ✅ Embedded & updated: ${item.documentId}`);
        successCount++;
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
    `🎉 Finished batch for ${modelName}: embedded ${successCount}/${data.length}`,
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

  // 4. Process Videos
  await processModel("videos", ["author", "location"], (item) => ({
    description: item.description,
    visibility: item.visibility,
    latitude: item.latitude,
    longitude: item.longitude,
    author: item.author
      ? {
          documentId: item.author.documentId,
          handle: item.author.handle,
          title: item.author.title,
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
  }));

  console.log("Finished Embedding Job successfully.");
  return { statusCode: 200, body: "Success" };
};
