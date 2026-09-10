# lambda-embed-content

Polls Strapi (jobs, properties, providers, videos, profiles) and Medusa (shop
products + shops) for rows without an embedding, embeds them with
`gemini-embedding-001` (768 dims, normalised) on Vertex AI, and writes the vector
back. Runs every 5 minutes from EventBridge; deployed by
`.github/workflows/deploy.yaml` as `LambdaEmbedContentStack-<env>`.

## Video hash contract

Videos differ from the other Strapi models (recombee-feed-remediation-plan.md §5 E5):

- `GET /videos/unembedded?limit=50` returns `[{documentId, text, hash}]`. The text
  (description + hashtags + transcript_text) and `hash` (= `embedding_desired_hash`)
  are built **server-side** by Strapi's shared builder; the Lambda embeds `text`
  verbatim instead of running its own mapper.
- `POST /videos/:documentId/embedding {embedding, hash}` applies the vector with
  `UPDATE … WHERE embedding_desired_hash = hash`. A **409** means the desired text
  changed while the Lambda was embedding: the write is skipped (logged as
  "stale"), the previous vector is kept, and the row is picked up again on the
  next poll with the new text. Stale rows are not failures.
- Rows selected by Strapi: `embedding IS NULL OR embedding_source_hash IS DISTINCT
  FROM embedding_desired_hash` (and a non-null desired hash), so editing a
  description or adding a transcript re-embeds automatically.

Backward compatibility: if a row arrives **without** `text` (a Strapi that
predates the contract) the Lambda falls back to the old in-Lambda video mapper
and posts `{embedding}` without a hash, which Strapi applies unconditionally.
Jobs, properties, providers and profiles are unchanged (no hash).

## Environment / secrets

`STRAPI_API_URL`, `STRAPI_API_TOKEN_SECRET_ARN` (`<env>/strapi/api-token`),
`VERTEX_CREDENTIALS_SECRET_ARN` (`<env>/vertex/credentials`),
`VERTEX_AI_PROJECT_ID`, `VERTEX_AI_LOCATION`, optional `MEDUSA_API_URL` +
`MEDUSA_KEYS_SECRET_ARN` (`<env>/expath-medusa-keys`).
