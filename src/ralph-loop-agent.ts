import {
  generateText,
  streamText,
  stepCountIs,
  type GenerateTextResult,
  type StreamTextResult,
  type ToolSet,
  type LanguageModel,
  type StopCondition,
} from 'ai';
import type { ModelMessage } from '@ai-sdk/provider-utils';
import type { RalphLoopAgentSettings } from './ralph-loop-agent-settings';
import type { VerifyCompletionResult } from './ralph-loop-agent-evaluator';

/**
 * Parameters for calling a RalphLoopAgent.
 */
export type RalphLoopAgentCallParameters = {
  /**
   * The prompt/task to complete.
   */
  prompt: string;

  /**
   * Abort signal for cancellation.
   */
  abortSignal?: AbortSignal;
};

/**
 * Result of a RalphLoopAgent execution.
 */
export interface RalphLoopAgentResult<TOOLS extends ToolSet = {}> {
  /**
   * The final text output.
   */
  readonly text: string;

  /**
   * The number of iterations that were executed.
   */
  readonly iterations: number;

  /**
   * Why the loop stopped.
   */
  readonly completionReason: 'verified' | 'max-iterations' | 'aborted';

  /**
   * The reason message from verifyCompletion (if provided).
   */
  readonly reason?: string;

  /**
   * The full result from the last iteration.
   */
  readonly result: GenerateTextResult<TOOLS, never>;

  /**
   * All results from each iteration.
   */
  readonly allResults: Array<GenerateTextResult<TOOLS, never>>;
}

/**
 * Stop condition for iteration count.
 */
export type IterationStopCondition = {
  type: 'iteration-count';
  count: number;
};

/**
 * Creates a stop condition that stops after N iterations.
 */
export function iterationCountIs(count: number): IterationStopCondition {
  return { type: 'iteration-count', count };
}

/**
 * A Ralph Loop Agent implements the "Ralph Wiggum" technique - an iterative
 * approach that continuously runs until a task is completed.
 *
 * The agent has two nested loops:
 * 1. **Outer loop (Ralph loop)**: Runs iterations until verifyCompletion returns true
 * 2. **Inner loop (Tool loop)**: Executes tools and LLM calls within each iteration
 *
 * @example
 * ```typescript
 * const agent = new RalphLoopAgent({
 *   model: 'anthropic/claude-opus-4.5',
 *   instructions: 'You are a helpful assistant.',
 *   tools: { readFile, writeFile },
 *   stopWhen: iterationCountIs(10),
 *   verifyCompletion: async ({ result }) => ({
 *     complete: result.text.includes('DONE'),
 *     reason: 'Task completed',
 *   }),
 * });
 *
 * const result = await agent.loop({ prompt: 'Do the task' });
 * ```
 */
export class RalphLoopAgent<TOOLS extends ToolSet = {}> {
  readonly version = 'ralph-agent-v1';

  private readonly settings: RalphLoopAgentSettings<TOOLS>;

  constructor(settings: RalphLoopAgentSettings<TOOLS>) {
    this.settings = settings;
  }

  /**
   * The id of the agent.
   */
  get id(): string | undefined {
    return this.settings.id;
  }

  /**
   * The tools that the agent can use.
   */
  get tools(): TOOLS {
    return this.settings.tools as TOOLS;
  }

  /**
   * The maximum number of iterations.
   */
  get maxIterations(): number {
    const stopWhen = this.settings.stopWhen;
    if (stopWhen && 'type' in stopWhen && stopWhen.type === 'iteration-count') {
      return stopWhen.count;
    }
    return 10; // default
  }

  /**
   * Runs the agent loop until completion or max iterations.
   */
  async loop({
    prompt,
    abortSignal,
  }: RalphLoopAgentCallParameters): Promise<RalphLoopAgentResult<TOOLS>> {
    const allResults: Array<GenerateTextResult<TOOLS, never>> = [];
    let currentMessages: Array<ModelMessage> = [];
    let iteration = 0;
    let completionReason: RalphLoopAgentResult<TOOLS>['completionReason'] = 'max-iterations';
    let reason: string | undefined;

    const maxIterations = this.maxIterations;

    // Build the initial user message
    const initialUserMessage: ModelMessage = {
      role: 'user',
      content: [{ type: 'text', text: prompt }],
    };

    // Add instructions as system message if provided
    const systemMessages = this.buildSystemMessages();

    while (iteration < maxIterations) {
      // Check for abort
      if (abortSignal?.aborted) {
        completionReason = 'aborted';
        break;
      }

      iteration++;
      const startTime = Date.now();

      // Call onIterationStart
      await this.settings.onIterationStart?.({ iteration });

      // Build messages for this iteration
      const messages: Array<ModelMessage> = [
        ...systemMessages,
        initialUserMessage,
        ...currentMessages,
      ];

      // If not the first iteration, add continuation prompt
      if (iteration > 1) {
        messages.push({
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Continue working on the task. The previous attempt was not complete.',
            },
          ],
        });
      }

      // Run the inner tool loop
      const result = (await generateText({
        model: this.settings.model,
        messages,
        tools: this.settings.tools,
        toolChoice: this.settings.toolChoice,
        stopWhen: this.settings.toolStopWhen ?? stepCountIs(20),
        maxOutputTokens: this.settings.maxOutputTokens,
        temperature: this.settings.temperature,
        topP: this.settings.topP,
        topK: this.settings.topK,
        presencePenalty: this.settings.presencePenalty,
        frequencyPenalty: this.settings.frequencyPenalty,
        stopSequences: this.settings.stopSequences,
        seed: this.settings.seed,
        experimental_telemetry: this.settings.experimental_telemetry,
        activeTools: this.settings.activeTools,
        prepareStep: this.settings.prepareStep,
        experimental_repairToolCall: this.settings.experimental_repairToolCall,
        providerOptions: this.settings.providerOptions,
        experimental_context: this.settings.experimental_context,
        abortSignal,
      })) as GenerateTextResult<TOOLS, never>;

      allResults.push(result);

      // Add the response messages to conversation history
      currentMessages = [...currentMessages, ...result.response.messages];

      const duration = Date.now() - startTime;

      // Call onIterationEnd
      await this.settings.onIterationEnd?.({
        iteration,
        duration,
        result,
      });

      // Verify completion
      if (this.settings.verifyCompletion) {
        const verification = await this.settings.verifyCompletion({
          result,
          iteration,
          allResults,
          originalPrompt: prompt,
        });

        if (verification.complete) {
          completionReason = 'verified';
          reason = verification.reason;
          break;
        }

        // If verification provides feedback, add it
        if (verification.reason && !verification.complete) {
          currentMessages.push({
            role: 'user',
            content: [
              {
                type: 'text',
                text: `Feedback: ${verification.reason}`,
              },
            ],
          });
        }
      }
    }

    const finalResult = allResults[allResults.length - 1]!;

    return {
      text: finalResult.text,
      iterations: iteration,
      completionReason,
      reason,
      result: finalResult,
      allResults,
    };
  }

  /**
   * Streams the agent loop. Streams only the final iteration.
   * For full control, use loop() with callbacks instead.
   */
  async stream({
    prompt,
    abortSignal,
  }: RalphLoopAgentCallParameters): Promise<StreamTextResult<TOOLS, never>> {
    const allResults: Array<GenerateTextResult<TOOLS, never>> = [];
    let currentMessages: Array<ModelMessage> = [];
    let iteration = 0;

    const maxIterations = this.maxIterations;

    const initialUserMessage: ModelMessage = {
      role: 'user',
      content: [{ type: 'text', text: prompt }],
    };

    const systemMessages = this.buildSystemMessages();

    // Run non-streaming iterations until completion or second-to-last
    while (iteration < maxIterations - 1) {
      if (abortSignal?.aborted) {
        break;
      }

      iteration++;
      const startTime = Date.now();

      await this.settings.onIterationStart?.({ iteration });

      const messages: Array<ModelMessage> = [
        ...systemMessages,
        initialUserMessage,
        ...currentMessages,
      ];

      if (iteration > 1) {
        messages.push({
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Continue working on the task. The previous attempt was not complete.',
            },
          ],
        });
      }

      const result = (await generateText({
        model: this.settings.model,
        messages,
        tools: this.settings.tools,
        toolChoice: this.settings.toolChoice,
        stopWhen: this.settings.toolStopWhen ?? stepCountIs(20),
        maxOutputTokens: this.settings.maxOutputTokens,
        temperature: this.settings.temperature,
        topP: this.settings.topP,
        topK: this.settings.topK,
        presencePenalty: this.settings.presencePenalty,
        frequencyPenalty: this.settings.frequencyPenalty,
        stopSequences: this.settings.stopSequences,
        seed: this.settings.seed,
        experimental_telemetry: this.settings.experimental_telemetry,
        activeTools: this.settings.activeTools,
        prepareStep: this.settings.prepareStep,
        experimental_repairToolCall: this.settings.experimental_repairToolCall,
        providerOptions: this.settings.providerOptions,
        experimental_context: this.settings.experimental_context,
        abortSignal,
      })) as GenerateTextResult<TOOLS, never>;

      allResults.push(result);
      currentMessages = [...currentMessages, ...result.response.messages];

      const duration = Date.now() - startTime;
      await this.settings.onIterationEnd?.({ iteration, duration, result });

      if (this.settings.verifyCompletion) {
        const verification = await this.settings.verifyCompletion({
          result,
          iteration,
          allResults,
          originalPrompt: prompt,
        });

        if (verification.complete) {
          // Complete early - return a stream for the final message
          return streamText({
            model: this.settings.model,
            messages: [...systemMessages, initialUserMessage, ...currentMessages],
            tools: this.settings.tools,
            toolChoice: this.settings.toolChoice,
            stopWhen: this.settings.toolStopWhen ?? stepCountIs(20),
            maxOutputTokens: this.settings.maxOutputTokens,
            temperature: this.settings.temperature,
            abortSignal,
          }) as StreamTextResult<TOOLS, never>;
        }

        if (verification.reason) {
          currentMessages.push({
            role: 'user',
            content: [{ type: 'text', text: `Feedback: ${verification.reason}` }],
          });
        }
      }
    }

    // Stream the final iteration
    iteration++;
    const finalMessages: Array<ModelMessage> = [
      ...systemMessages,
      initialUserMessage,
      ...currentMessages,
    ];

    if (iteration > 1) {
      finalMessages.push({
        role: 'user',
        content: [
          {
            type: 'text',
            text: 'Continue working on the task. The previous attempt was not complete.',
          },
        ],
      });
    }

    return streamText({
      model: this.settings.model,
      messages: finalMessages,
      tools: this.settings.tools,
      toolChoice: this.settings.toolChoice,
      stopWhen: this.settings.toolStopWhen ?? stepCountIs(20),
      maxOutputTokens: this.settings.maxOutputTokens,
      temperature: this.settings.temperature,
      topP: this.settings.topP,
      topK: this.settings.topK,
      presencePenalty: this.settings.presencePenalty,
      frequencyPenalty: this.settings.frequencyPenalty,
      stopSequences: this.settings.stopSequences,
      seed: this.settings.seed,
      experimental_telemetry: this.settings.experimental_telemetry,
      activeTools: this.settings.activeTools,
      prepareStep: this.settings.prepareStep,
      experimental_repairToolCall: this.settings.experimental_repairToolCall,
      providerOptions: this.settings.providerOptions,
      experimental_context: this.settings.experimental_context,
      abortSignal,
    }) as StreamTextResult<TOOLS, never>;
  }

  /**
   * Build system messages from instructions.
   */
  private buildSystemMessages(): Array<ModelMessage> {
    const { instructions } = this.settings;

    if (!instructions) {
      return [];
    }

    if (typeof instructions === 'string') {
      return [{ role: 'system', content: instructions }];
    }

    if (Array.isArray(instructions)) {
      return instructions;
    }

    return [instructions];
  }
}
