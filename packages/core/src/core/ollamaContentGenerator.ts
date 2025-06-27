import ollama, { Ollama } from 'ollama';
import {
  GenerateContentParameters,
  GenerateContentResponse,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  Content,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';

export class OllamaContentGenerator implements ContentGenerator {
  private client: Ollama
  constructor(host?: string) {
    this.client = host ? new Ollama({ host }) : (ollama as unknown as Ollama)
  }

  private contentsToMessages(contents: Content[]) {
    return contents.map((c) => {
      const text = c.parts?.map((p: any) => p.text ?? '').join('') ?? ''
      const role = c.role === 'user' ? 'user' : 'assistant'
      return { role, content: text }
    })
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const messages = this.contentsToMessages(request.contents as Content[])
    const response = await this.client.chat({
      model: request.model,
      messages,
    })
    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [{ text: (response as any).message.content }],
          } as Content,
        },
      ],
    } as GenerateContentResponse
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = this.contentsToMessages(request.contents as Content[])
    const stream = await this.client.chat({
      model: request.model,
      messages,
      stream: true,
    })
    const gen = async function* (s: AsyncGenerator<any>): AsyncGenerator<GenerateContentResponse> {
      for await (const part of s) {
        yield {
          candidates: [
            {
              content: {
                role: 'model',
                parts: [{ text: part.message.content }],
              } as Content,
            },
          ],
        } as GenerateContentResponse
      }
    }
    return gen(stream as AsyncGenerator<any>)
  }

  async countTokens(_: CountTokensParameters): Promise<CountTokensResponse> {
    return { totalTokens: 0 }
  }

  async embedContent(_: EmbedContentParameters): Promise<EmbedContentResponse> {
    return { embeddings: [] }
  }
}
