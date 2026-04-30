#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { LambdaEmbedContentStack } from "../lib/infra-stack";

const app = new cdk.App();

// Read deployment environment and image URI from environment variables or context
const deploymentEnv = app.node.tryGetContext("env") || process.env.DEPLOYMENT_ENV || "test";
const imageUri = app.node.tryGetContext("imageUri") || process.env.IMAGE_URI;

if (!imageUri) {
  throw new Error("Missing imageUri context or environment variable");
}

new LambdaEmbedContentStack(app, `LambdaEmbedContentStack-${deploymentEnv}`, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || "ap-southeast-2",
  },
  deploymentEnv,
  imageUri,
});
