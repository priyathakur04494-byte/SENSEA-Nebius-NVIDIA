/**
 * BHASHINI API CLIENT
 * ---------------------
 * Bhashini's translation flow is a two-step handshake:
 *   1. Ask `getModelsPipeline` which translation model/endpoint to use for
 *      a given source→target language pair, and get a short-lived
 *      inference API key back.
 *   2. Call that model's own inference endpoint with the actual text.
 *
 * Step 1 responses are cacheable per language pair (they don't change
 * often), so we cache them in memory to avoid re-negotiating on every
 * single translation call.
 *
 * NOTE: Bhashini's exact request/response shape has changed between API
 * versions in the past. This follows their documented v0 pipeline shape;
 * if calls start failing with a schema error, check
 * https://bhashini.gitbook.io/bhashini-apis for the current contract.
 */

import { BHASHINI_CONFIG, isBhashiniConfigured } from "@/config/bhashiniConfig";
import { SupportedLanguage } from "@/types";

interface PipelineInfo {
  callbackUrl: string;
  inferenceApiKey: string;
  serviceId: string;
}

const pipelineCache = new Map<string, PipelineInfo>();

async function getPipeline(
  sourceLanguage: SupportedLanguage,
  targetLanguage: SupportedLanguage
): Promise<PipelineInfo> {
  const cacheKey = `${sourceLanguage}-${targetLanguage}`;
  const cached = pipelineCache.get(cacheKey);
  if (cached) return cached;

  const response = await fetch(BHASHINI_CONFIG.pipelineEndpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      userID: BHASHINI_CONFIG.userID,
      ulcaApiKey: BHASHINI_CONFIG.ulcaApiKey,
    },
    body: JSON.stringify({
      pipelineTasks: [
        {
          taskType: "translation",
          config: {
            language: { sourceLanguage, targetLanguage },
          },
        },
      ],
      pipelineRequestConfig: { pipelineId: BHASHINI_CONFIG.pipelineId },
    }),
  });

  if (!response.ok) {
    throw new Error(`Bhashini pipeline request failed: ${response.status}`);
  }

  const data = await response.json();
  const endpoint = data.pipelineInferenceAPIEndPoint;
  const task = data.pipelineResponseConfig?.[0]?.config?.[0];

  if (!endpoint || !task) {
    throw new Error("Bhashini pipeline response missing expected fields");
  }

  const info: PipelineInfo = {
    callbackUrl: endpoint.callbackUrl,
    inferenceApiKey: endpoint.inferenceApiKey.value,
    serviceId: task.serviceId,
  };

  pipelineCache.set(cacheKey, info);
  return info;
}

/**
 * Translates text using Bhashini. Throws on any failure — callers should
 * catch and fall back to the mock dictionary (see translationService.ts)
 * rather than let a network error crash the caption flow.
 */
export async function bhashiniTranslate(
  text: string,
  sourceLanguage: SupportedLanguage,
  targetLanguage: SupportedLanguage
): Promise<string> {
  if (!isBhashiniConfigured()) {
    throw new Error("Bhashini credentials not configured — see src/config/bhashiniConfig.ts");
  }

  const pipeline = await getPipeline(sourceLanguage, targetLanguage);

  const response = await fetch(pipeline.callbackUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: pipeline.inferenceApiKey,
    },
    body: JSON.stringify({
      pipelineTasks: [
        {
          taskType: "translation",
          config: {
            language: { sourceLanguage, targetLanguage },
            serviceId: pipeline.serviceId,
          },
        },
      ],
      inputData: {
        input: [{ source: text }],
      },
    }),
  });

  if (!response.ok) {
    throw new Error(`Bhashini inference request failed: ${response.status}`);
  }

  const data = await response.json();
  const translated = data.pipelineResponse?.[0]?.output?.[0]?.target;

  if (!translated) {
    throw new Error("Bhashini inference response missing translated text");
  }

  return translated;
}
