import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import * as iam from "aws-cdk-lib/aws-iam";
import * as ecr from "aws-cdk-lib/aws-ecr";

export interface LambdaEmbedContentStackProps extends cdk.StackProps {
  deploymentEnv: string; // "test" | "prod"
  imageUri: string; // E.g., "123456789012.dkr.ecr.ap-southeast-2.amazonaws.com/lambda-embed-content:latest"
}

export class LambdaEmbedContentStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: LambdaEmbedContentStackProps) {
    super(scope, id, props);

    // Fetch secrets securely
    const strapiSecret = secretsmanager.Secret.fromSecretNameV2(
      this,
      "StrapiApiToken",
      `${props.deploymentEnv}/strapi/api-token`
    );

    const vertexCredentialsSecret = secretsmanager.Secret.fromSecretNameV2(
      this,
      "VertexCredentials",
      `${props.deploymentEnv}/vertex/credentials`
    );

    const strapiUrl = `https://cms-v2.${props.deploymentEnv}.expath.com.au/api`;

    // Format: <accountId>.dkr.ecr.<region>.amazonaws.com/<repoName>:<tag>
    const match = props.imageUri.match(/^(\d+)\.dkr\.ecr\.[^.]+\.amazonaws\.com\/([^:]+):(.*)$/);
    if (!match) {
      throw new Error(`Invalid imageUri format: ${props.imageUri}`);
    }
    const accountId = match[1];
    const repositoryName = match[2];
    const tag = match[3];

    const repository = ecr.Repository.fromRepositoryAttributes(this, "EcrRepo", {
      repositoryArn: `arn:aws:ecr:${cdk.Stack.of(this).region}:${accountId}:repository/${repositoryName}`,
      repositoryName: repositoryName,
    });

    // Create the Lambda function from ECR Image
    const embedLambda = new lambda.DockerImageFunction(this, "EmbedContentLambda", {
      code: lambda.DockerImageCode.fromEcr(repository, { tagOrDigest: tag }),
      memorySize: 1024,
      timeout: cdk.Duration.minutes(10), // Batch processing needs time
      environment: {
        STRAPI_API_URL: strapiUrl,
        VERTEX_AI_PROJECT_ID: "expath-app",
        VERTEX_AI_LOCATION: "australia-southeast1",
        // The Lambda will read these secrets at runtime
      },
    });

    // Grant Lambda access to read secrets
    strapiSecret.grantRead(embedLambda);
    vertexCredentialsSecret.grantRead(embedLambda);

    // We pass the secret ARNs as env vars so the lambda code knows where to fetch them
    embedLambda.addEnvironment("STRAPI_API_TOKEN_SECRET_ARN", strapiSecret.secretArn);
    embedLambda.addEnvironment("VERTEX_CREDENTIALS_SECRET_ARN", vertexCredentialsSecret.secretArn);

    // Give Lambda permission to call Secrets Manager GetSecretValue (grantRead covers it, but ensuring explicit access if needed)
    embedLambda.addToRolePolicy(new iam.PolicyStatement({
      actions: ["secretsmanager:GetSecretValue"],
      resources: [strapiSecret.secretArn, vertexCredentialsSecret.secretArn],
    }));

    // Trigger every 5 minutes from EventBridge
    const rule = new events.Rule(this, "ScheduleRule", {
      schedule: events.Schedule.rate(cdk.Duration.minutes(5)),
    });

    rule.addTarget(new targets.LambdaFunction(embedLambda));
  }
}
