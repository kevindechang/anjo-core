import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import test from 'node:test';

import {
  DEFAULT_APPRAISAL_GOALS,
  applyLengthFactor,
  appraiseInput,
  appraiseTurn,
  baselineWeight,
  buildPresenceVector,
  candidateScore,
  cleanText,
  decodingParams,
  decayMood,
  expectationEmotions,
  isAmbivalent,
  lengthFactor,
  moodCongruenceFactor,
  moodInertia,
  moodOctant,
  occCarryDecay,
  presenceLine,
  recencyWeight,
  recencyWeightFromTimestamp,
  similarityFromDistance,
  stageInt,
  stateEmotions,
  type AppraisalGoals,
  type PadMood,
  type Personality,
} from '../src/index.js';

type JsonCase = { in: any; out: any };
type CaseGroup = Record<string, JsonCase[]>;
interface Corpus {
  _meta: { cases: number; affect_habituation: boolean };
  affect_control: CaseGroup;
  retrieval: CaseGroup;
  appraisal: CaseGroup;
  surfacing: CaseGroup;
}

const corpus = JSON.parse(
  readFileSync(resolve(process.cwd(), '..', 'shared', 'golden', 'kernel_golden.json'), 'utf8'),
) as Corpus;

const EPSILON = 1e-9;
const RECENCY_EPSILON = 1e-9;

function approx(actual: number, expected: number, epsilon = EPSILON): void {
  assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} != ${expected} (±${epsilon})`);
}

function approxMap(actual: Record<string, number>, expected: Record<string, number>): void {
  assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort());
  for (const [key, value] of Object.entries(expected)) approx(actual[key] as number, value);
}

function approxValue(actual: unknown, expected: unknown): void {
  if (typeof expected === 'number') {
    approx(actual as number, expected);
    return;
  }
  if (expected === null || typeof expected !== 'object') {
    assert.equal(actual, expected);
    return;
  }
  if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual));
    assert.equal(actual.length, expected.length);
    expected.forEach((value, index) => approxValue(actual[index], value));
    return;
  }
  assert.ok(actual !== null && typeof actual === 'object');
  const actualRecord = actual as Record<string, unknown>;
  const expectedRecord = expected as Record<string, unknown>;
  assert.deepEqual(Object.keys(actualRecord).sort(), Object.keys(expectedRecord).sort());
  for (const key of Object.keys(expectedRecord)) approxValue(actualRecord[key], expectedRecord[key]);
}

function mood(value: any): PadMood | null {
  return value === null
    ? null
    : { valence: value.valence, arousal: value.arousal, dominance: value.dominance };
}

test('TypeScript reproduces all 225 non-habituating kernel vectors', () => {
  assert.equal(corpus._meta.cases, 225);
  assert.equal(corpus._meta.affect_habituation, false);
  let exercised = 0;

  const affect = corpus.affect_control;
  for (const vector of affect.mood_octant ?? []) {
    assert.equal(moodOctant(vector.in.valence, vector.in.arousal, vector.in.dominance), vector.out);
    exercised += 1;
  }
  for (const vector of affect.decoding_params ?? []) {
    const actual = decodingParams(mood(vector.in));
    approx(actual.temperature, vector.out.temperature);
    if (vector.out.top_p === null) assert.equal(actual.topP, null);
    else approx(actual.topP as number, vector.out.top_p);
    exercised += 1;
  }
  for (const vector of affect.length_factor ?? []) {
    approx(lengthFactor(mood(vector.in)), vector.out);
    exercised += 1;
  }
  for (const vector of affect.apply_length_factor ?? []) {
    const inputMood = mood(vector.in) as PadMood;
    assert.equal(applyLengthFactor(vector.in.base_tokens, inputMood), vector.out);
    exercised += 1;
  }
  for (const vector of affect.is_ambivalent ?? []) {
    assert.equal(isAmbivalent(vector.in), vector.out);
    exercised += 1;
  }

  const retrieval = corpus.retrieval;
  for (const vector of retrieval.recency_weight ?? []) {
    if ('timestamp' in vector.in) {
      approx(recencyWeightFromTimestamp(vector.in.timestamp), vector.out);
    } else {
      approx(recencyWeight(vector.in.days_ago), vector.out, RECENCY_EPSILON);
    }
    exercised += 1;
  }
  for (const vector of retrieval.mood_congruence_factor ?? []) {
    approx(
      moodCongruenceFactor(
        vector.in.mem_valence,
        vector.in.mood_valence,
        vector.in.congruence_on,
      ),
      vector.out,
    );
    exercised += 1;
  }
  for (const vector of retrieval.candidate_score ?? []) {
    approx(
      candidateScore({
        similarity: similarityFromDistance(vector.in.distance),
        recency: recencyWeight(vector.in.days_ago),
        episode: vector.in.episode,
        significance: vector.in.significance,
        recallCount: vector.in.recall_count,
      }),
      vector.out,
    );
    exercised += 1;
  }

  const appraisal = corpus.appraisal;
  for (const vector of appraisal.mood_inertia ?? []) {
    approx(moodInertia(vector.in as Personality), vector.out);
    exercised += 1;
  }
  for (const vector of appraisal.stage_int ?? []) {
    assert.equal(stageInt(vector.in.stage), vector.out);
    exercised += 1;
  }
  for (const vector of appraisal.baseline_weight ?? []) {
    approx(baselineWeight(vector.in.stage_int), vector.out);
    exercised += 1;
  }
  for (const vector of appraisal.decay_mood ?? []) {
    const input = vector.in;
    approxValue(
      decayMood(
        { valence: input.valence, arousal: input.arousal, dominance: input.dominance },
        { O: input.O, C: input.C, E: input.E, A: input.A, N: input.N },
        input.stage_int,
        input.baseline_valence,
      ),
      vector.out,
    );
    exercised += 1;
  }
  for (const vector of appraisal.appraise_input ?? []) {
    const input = vector.in;
    const goals: AppraisalGoals = {
      rapport: input.rapport,
      intellectual: input.intellectual,
      autonomy: input.autonomy,
      respect: input.respect,
      honesty: input.honesty,
    };
    const actual = appraiseInput(
      { valence: input.valence, arousal: input.arousal, dominance: input.dominance },
      goals,
      input.intent,
      input.baseline_valence,
    );
    approxMap(actual.emotions, vector.out.emotions);
    approxValue(actual.mood, vector.out.mood);
    approx(actual.baselineValence, vector.out.baseline_valence);
    exercised += 1;
  }
  for (const vector of appraisal.expectation_emotions ?? []) {
    approxMap(expectationEmotions(vector.in.expectation, vector.in.message), vector.out);
    exercised += 1;
  }
  for (const vector of appraisal.occ_carry_decay ?? []) {
    approxMap(occCarryDecay(vector.in), vector.out);
    exercised += 1;
  }
  for (const vector of appraisal.state_emotions ?? []) {
    approxMap(
      stateEmotions(
        { valence: vector.in.valence, arousal: vector.in.arousal, dominance: 0 },
        vector.in.longing,
      ),
      vector.out,
    );
    exercised += 1;
  }
  for (const vector of appraisal.appraise_turn ?? []) {
    const input = vector.in;
    const actual = appraiseTurn({
      mood: { valence: input.valence, arousal: input.arousal, dominance: input.dominance },
      personality: { O: input.O, C: input.C, E: input.E, A: input.A, N: input.N },
      goals: DEFAULT_APPRAISAL_GOALS,
      stageInt: input.stage_int,
      baselineValence: input.baseline_valence,
      attachmentLonging: input.longing,
      intent: input.intent,
      occCarry: input.occ_carry,
      expectation: input.expectation,
      message: input.message,
    });
    approxValue(actual.mood, vector.out.mood);
    approxMap(actual.activeEmotions, vector.out.active_emotions);
    approxMap(actual.occCarry, vector.out.occ_carry);
    approx(actual.baselineValence, vector.out.baseline_valence);
    exercised += 1;
  }

  const surfacing = corpus.surfacing;
  for (const vector of surfacing.clean_text ?? []) {
    assert.equal(cleanText(vector.in.value, vector.in.max_len), vector.out);
    exercised += 1;
  }
  for (const vector of surfacing.presence_line ?? []) {
    assert.equal(
      presenceLine({
        reflectionPending: vector.in.reflection_pending,
        dueIntention: vector.in.due_intention,
        carriedThought: vector.in.carried_thought,
        openThread: vector.in.open_thread,
      }),
      vector.out,
    );
    exercised += 1;
  }
  for (const vector of surfacing.presence_vector ?? []) {
    const input = vector.in;
    const actual = buildPresenceVector(
      {
        mood: { valence: input.valence, arousal: input.arousal, dominance: input.dominance },
        attachment: { longing: input.longing, comfort: input.comfort },
        relationship: { stage: input.stage, trustScore: input.trust_score },
        carriedThought: input.carried_thought,
      },
      {
        reflectionPending: input.cognition.reflection_pending,
        dueIntention: input.cognition.due_intention,
        openThread: input.cognition.open_thread,
        intentionality: input.cognition.intentionality,
        curiosity: input.cognition.curiosity,
      },
    );
    approxValue(actual, vector.out);
    exercised += 1;
  }

  assert.equal(exercised, corpus._meta.cases);
});
