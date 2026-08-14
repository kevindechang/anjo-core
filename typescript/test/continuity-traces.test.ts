import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import test from 'node:test';

import {
  DEFAULT_APPRAISAL_GOALS,
  DEFAULT_APPRAISAL_POLICY,
  type PadMood,
  type Personality,
} from '../src/index.js';

interface ExpectedStep {
  event: { intent: string; message: string };
  expected: {
    mood: PadMood;
    baseline_valence: number;
    active_emotions: Record<string, number>;
    occ_carry: Record<string, number>;
  };
}

interface ContinuityTrace {
  name: string;
  initial: {
    mood: PadMood;
    personality: Personality;
    baseline_valence: number;
  };
  steps: ExpectedStep[];
}

interface ContinuityCorpus {
  _meta: { traces: number };
  traces: ContinuityTrace[];
}

const corpus = JSON.parse(readFileSync(
  resolve(process.cwd(), '..', 'shared', 'golden', 'continuity_traces.json'),
  'utf8',
)) as ContinuityCorpus;

function approx(actual: number, expected: number, location: string): void {
  assert.ok(Math.abs(actual - expected) <= 1e-9, `${location}: ${actual} != ${expected}`);
}

function approxMap(
  actual: Readonly<Record<string, number>>,
  expected: Readonly<Record<string, number>>,
  location: string,
): void {
  assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), `${location}: keys`);
  for (const [key, expectedValue] of Object.entries(expected)) {
    approx(actual[key] as number, expectedValue, `${location}.${key}`);
  }
}

test('default appraisal policy reproduces all longitudinal continuity traces', () => {
  assert.equal(corpus._meta.traces, 3);
  assert.equal(corpus.traces.length, corpus._meta.traces);

  for (const trace of corpus.traces) {
    let mood = trace.initial.mood;
    let baselineValence = trace.initial.baseline_valence;
    let occCarry: Readonly<Record<string, number>> = {};

    for (const [index, step] of trace.steps.entries()) {
      const actual = DEFAULT_APPRAISAL_POLICY({
        mood,
        personality: trace.initial.personality,
        goals: DEFAULT_APPRAISAL_GOALS,
        stageInt: 1,
        baselineValence,
        attachmentLonging: 0,
        intent: step.event.intent,
        occCarry,
        expectation: '',
        message: step.event.message,
      });
      const location = `${trace.name} step ${index + 1}`;
      approx(actual.mood.valence, step.expected.mood.valence, `${location}: mood.valence`);
      approx(actual.mood.arousal, step.expected.mood.arousal, `${location}: mood.arousal`);
      approx(actual.mood.dominance, step.expected.mood.dominance, `${location}: mood.dominance`);
      approx(actual.baselineValence, step.expected.baseline_valence, `${location}: baseline`);
      approxMap(
        actual.activeEmotions,
        step.expected.active_emotions,
        `${location}: active emotions`,
      );
      approxMap(actual.occCarry, step.expected.occ_carry, `${location}: OCC carry`);

      mood = actual.mood;
      baselineValence = actual.baselineValence;
      occCarry = actual.occCarry;
    }
  }
});
