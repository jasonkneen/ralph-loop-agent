# ralph-wiggum

> **⚠️ EXPERIMENTAL - USE AT YOUR OWN RISK**
>
> This package is highly experimental. The iterative nature of the Ralph Wiggum technique can result in **high token usage and significant costs**. Each iteration consumes tokens, and the agent may run multiple iterations before completing a task. Monitor your usage carefully and set appropriate `stopWhen` limits.

An iterative AI agent that implements the "Ralph Wiggum" technique - continuously running until a task is completed.

## Installation

```bash
npm install ralph-wiggum ai zod
```

## What is the Ralph Wiggum Technique?

Named after the lovably persistent character from *The Simpsons*, the Ralph Wiggum technique is an iterative approach that keeps running an AI agent until a task is evaluated as complete. Unlike a standard tool loop that stops when the model stops calling tools, Ralph keeps trying until the job is done.

```
┌─────────────────────────────────────────────────┐
│  ┌─────────────────────────────────┐            │
│  │  Inner tool loop                │            │
│  │  (LLM ↔ tools until no more)    │            │
│  └─────────────────────────────────┘            │
│              ↓                                  │
│  verifyCompletion: "Is the TASK complete?"      │
│              ↓                                  │
│       No? → Run another iteration               │
│       Yes? → Return final result                │
└─────────────────────────────────────────────────┘
```

## Usage

### Basic Example

```typescript
import { RalphLoopAgent, iterationCountIs } from 'ralph-wiggum';

const agent = new RalphLoopAgent({
  model: 'anthropic/claude-opus-4.5',
  instructions: 'You are a helpful coding assistant.',
  stopWhen: iterationCountIs(10),
  verifyCompletion: async ({ result }) => ({
    complete: result.text.includes('DONE'),
    reason: 'Task completed successfully',
  }),
});

const { text, iterations, completionReason } = await agent.loop({
  prompt: 'Create a function that calculates fibonacci numbers',
});

console.log(text);
console.log(`Completed in ${iterations} iterations`);
console.log(`Reason: ${completionReason}`);
```

### Migration Example

```typescript
import { RalphLoopAgent, iterationCountIs } from 'ralph-wiggum';

const migrationAgent = new RalphLoopAgent({
  model: 'anthropic/claude-opus-4.5',
  instructions: `You are migrating a codebase from Jest to Vitest.
    
    Completion criteria:
    - All test files use vitest imports
    - vitest.config.ts exists
    - All tests pass when running 'pnpm test'`,
  
  tools: { readFile, writeFile, execute },
  
  stopWhen: iterationCountIs(50),
  
  verifyCompletion: async () => {
    const checks = await Promise.all([
      fileExists('vitest.config.ts'),
      !await fileExists('jest.config.js'),
      noFilesMatch('**/*.test.ts', /from ['"]@jest/),
      fileContains('package.json', '"vitest"'),
    ]);
    
    return { 
      complete: checks.every(Boolean),
      reason: checks.every(Boolean) ? 'Migration complete' : 'Structural checks failed'
    };
  },

  onIterationStart: ({ iteration }) => console.log(`Starting iteration ${iteration}`),
  onIterationEnd: ({ iteration, duration }) => console.log(`Iteration ${iteration} completed in ${duration}ms`),
});

const result = await migrationAgent.loop({
  prompt: 'Migrate all Jest tests to Vitest.',
});

console.log(result.text);
console.log(result.iterations);
console.log(result.completionReason);
```

### With Tools

```typescript
import { RalphLoopAgent, iterationCountIs } from 'ralph-wiggum';
import { tool } from 'ai';
import { z } from 'zod';

const agent = new RalphLoopAgent({
  model: 'anthropic/claude-opus-4.5',
  instructions: 'You help users with file operations.',
  tools: {
    readFile: tool({
      description: 'Read a file from disk',
      parameters: z.object({ path: z.string() }),
      execute: async ({ path }) => ({ content: '...' }),
    }),
    writeFile: tool({
      description: 'Write content to a file',
      parameters: z.object({ path: z.string(), content: z.string() }),
      execute: async ({ path, content }) => ({ success: true }),
    }),
  },
  stopWhen: iterationCountIs(10),
  verifyCompletion: ({ result }) => ({
    complete: result.text.includes('All files updated'),
  }),
});
```

### Streaming

```typescript
const stream = await agent.stream({
  prompt: 'Build a calculator',
});

for await (const chunk of stream.textStream) {
  process.stdout.write(chunk);
}
```

Note: Streaming runs non-streaming iterations until verification passes or the final iteration, then streams that last iteration.

## API Reference

### `RalphLoopAgent`

#### Constructor Options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `model` | `LanguageModel` | ✅ | - | The AI model (AI Gateway string format) |
| `instructions` | `string` | ❌ | - | System prompt for the agent |
| `tools` | `ToolSet` | ❌ | - | Tools the agent can use |
| `stopWhen` | `IterationStopCondition` | ❌ | `iterationCountIs(10)` | When to stop the outer loop |
| `toolStopWhen` | `StopCondition` | ❌ | `stepCountIs(20)` | When to stop the inner tool loop |
| `verifyCompletion` | `function` | ❌ | - | Function to verify task completion |
| `onIterationStart` | `function` | ❌ | - | Called at start of each iteration |
| `onIterationEnd` | `function` | ❌ | - | Called at end of each iteration |

#### `iterationCountIs(n)`

Creates a stop condition that stops after `n` iterations.

```typescript
import { iterationCountIs } from 'ralph-wiggum';

stopWhen: iterationCountIs(50)
```

#### `verifyCompletion`

Function to verify if the task is complete:

```typescript
verifyCompletion: async ({ result, iteration, allResults, originalPrompt }) => ({
  complete: boolean,
  reason?: string, // Feedback if not complete, or explanation if complete
})
```

#### Methods

**`loop(options)`** - Runs the agent loop until completion
- Returns: `RalphLoopAgentResult`

```typescript
interface RalphLoopAgentResult {
  text: string;                              // Final output text
  iterations: number;                        // Number of iterations run
  completionReason: 'verified' | 'max-iterations' | 'aborted';
  reason?: string;                           // Reason from verifyCompletion
  result: GenerateTextResult;               // Full result from last iteration
  allResults: GenerateTextResult[];         // All iteration results
}
```

**`stream(options)`** - Streams the final iteration
- Returns: `StreamTextResult`

## License

Apache-2.0
