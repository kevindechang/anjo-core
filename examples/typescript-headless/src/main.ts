import {
  CompanionEngine,
  InMemoryRetriever,
  InMemoryStore,
  ScriptedModelAdapter,
} from '@anjo-ai/companion-core';

const store = new InMemoryStore({
  state: { mood: { valence: 0.1, arousal: 0.05, dominance: 0 } },
});
const model = new ScriptedModelAdapter({
  gates: [
    { intent: 'CURIOSITY', shouldRespond: true, shouldRetrieve: true },
    { intent: 'VULNERABILITY', shouldRespond: true, shouldRetrieve: false },
    { intent: 'CURIOSITY', shouldRespond: true, shouldRetrieve: false },
  ],
  responses: [
    ['That thread still has some pull. ', 'A fresh attempt can change its shape.'],
    ['A rough page can still be evidence that you returned.'],
    ['Wanting to try tomorrow is the part worth carrying forward.'],
  ],
});
const retriever = new InMemoryRetriever([
  {
    id: 'demo-memory',
    text: 'The user has been learning to sketch.',
    distance: 0.18,
    daysAgo: 2,
    significance: 0.6,
  },
]);
const engine = new CompanionEngine({
  model,
  store,
  retriever,
  instruction: 'Respond as a thoughtful fictional guide in this local demonstration.',
});

const turns = [
  'I picked up my sketchbook again.',
  'The first page looks terrible.',
  'Still, I want to try again tomorrow.',
] as const;

console.log('Initial presence:');
console.log(JSON.stringify(await engine.presence(), null, 2));
for (const [index, message] of turns.entries()) {
  let streamed = '';
  const result = await engine.turn(message, {
    onToken: (chunk) => { streamed += chunk; },
  });
  const evidence = result.memories.map((memory) => memory.id);
  const mood = result.mood;
  console.log(`\nTurn ${index + 1} — User: ${message}`);
  console.log(`Companion: ${streamed}`);
  console.log(
    `State: intent=${result.intent}, PAD=(${mood.valence.toFixed(4)}, `
      + `${mood.arousal.toFixed(4)}, ${mood.dominance.toFixed(4)}), `
      + `evidence=${evidence.length ? evidence.join(',') : 'none'}`,
  );
  const presence = await engine.presence();
  console.log(`Presence: ${presence.line} (${presence.mode})`);
}
