import { GoogleGenAI } from '@google/genai';
import { AuthType } from '@office-ai/aioncli-core';
import type { RotatingApiClientOptions } from './RotatingApiClient';
import { RotatingApiClient } from './RotatingApiClient';
import {
  OpenAI2GeminiConverter,
  type OpenAIChatCompletionParams,
  type OpenAIChatCompletionResponse,
} from './OpenAI2GeminiConverter';

export interface GeminiClientConfig {
  model?: string;
  baseURL?: string;
  requestOptions?: Record<string, unknown>;
}

export class GeminiRotatingClient extends RotatingApiClient<GoogleGenAI> {
  private readonly config: GeminiClientConfig;
  private readonly converter: OpenAI2GeminiConverter;

  constructor(
    apiKeys: string,
    config: GeminiClientConfig = {},
    options: RotatingApiClientOptions = {},
    authType: AuthType = AuthType.USE_GEMINI
  ) {
    const createClient = (apiKey: string) => {
      const cleanedApiKey = apiKey.replace(/[\s\r\n\t]/g, '').trim();
      const clientConfig: {
        apiKey?: string;
        vertexai: boolean;
        httpOptions?: { baseUrl?: string; headers?: Record<string, string> };
      } = {
        apiKey: cleanedApiKey === '' ? undefined : cleanedApiKey,
        vertexai: authType === AuthType.USE_VERTEX_AI,
      };
      if (config.baseURL) {
        // 自定义 baseUrl 通常指向 Bearer 鉴权的代理网关。SDK 默认只发 x-goog-api-key，
        // 这里额外注入 Authorization: Bearer 以兼容此类代理；同时保留 apiKey 字段，
        // 让走原生 Google API 的代理也能正常工作。
        const headers: Record<string, string> = {};
        if (cleanedApiKey) {
          headers.Authorization = `Bearer ${cleanedApiKey}`;
        }
        clientConfig.httpOptions = {
          baseUrl: config.baseURL,
          ...(Object.keys(headers).length > 0 && { headers }),
        };
      }
      return new GoogleGenAI(clientConfig);
    };

    super(apiKeys, authType, createClient, options);
    this.config = config;
    this.converter = new OpenAI2GeminiConverter({
      defaultModel: config.model || 'gemini-1.5-flash',
    });
  }

  protected getCurrentApiKey(): string | undefined {
    if (this.apiKeyManager?.hasMultipleKeys()) {
      // For Gemini, try to get from environment first
      return process.env.GEMINI_API_KEY || this.apiKeyManager.getCurrentKey();
    }
    // Use base class method for single key
    return super.getCurrentApiKey();
  }

  // Remove async override since base class is now sync
  // protected async initializeClient(): Promise<void> {
  //   await super.initializeClient();
  // }

  // Basic method for Gemini operations - can be extended as needed
  async generateContent(prompt: string, config?: Record<string, unknown>): Promise<unknown> {
    return await this.executeWithRetry(async (client) => {
      // client is GoogleGenAI, we need client.models to get the content generator
      const model = await client.models.generateContent({
        model: this.config.model || 'gemini-1.5-flash',
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        ...config,
      });
      return model;
    });
  }

  // OpenAI-compatible createChatCompletion method for unified interface
  async createChatCompletion(
    params: OpenAIChatCompletionParams,
    options?: { signal?: AbortSignal; timeout?: number }
  ): Promise<OpenAIChatCompletionResponse> {
    // Handle request cancellation
    if (options?.signal?.aborted) {
      throw new Error('Request was aborted');
    }

    return await this.executeWithRetry(async (client) => {
      // Convert OpenAI format to Gemini format using converter
      const geminiRequest = this.converter.convertRequest(params);

      // Call Gemini API
      const geminiResponse = await client.models.generateContent(geminiRequest);

      // Convert Gemini response back to OpenAI format using converter
      return this.converter.convertResponse(geminiResponse, params.model);
    });
  }
}
