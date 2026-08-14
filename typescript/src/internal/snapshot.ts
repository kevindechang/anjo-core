import type { DeepReadonly } from '../contracts.js';

function copy(value: unknown, seen: Map<object, unknown>): unknown {
  if (value === null || typeof value !== 'object') return value;
  const existing = seen.get(value);
  if (existing !== undefined) return existing;
  if (value instanceof Date) return new Date(value.getTime());
  if ('aborted' in value && typeof value.aborted === 'boolean') {
    const signal = value as { readonly aborted: boolean; readonly reason?: unknown };
    return {
      aborted: signal.aborted,
      ...('reason' in signal ? { reason: copy(signal.reason, seen) } : {}),
    };
  }
  if (Array.isArray(value)) {
    const result: unknown[] = [];
    seen.set(value, result);
    for (const item of value) result.push(copy(item, seen));
    return result;
  }
  const result: Record<string, unknown> = {};
  seen.set(value, result);
  for (const [key, item] of Object.entries(value)) result[key] = copy(item, seen);
  return result;
}

export function cloneValue<T>(value: T): T {
  return copy(value, new Map()) as T;
}

function freeze(value: unknown, seen: Set<object>): void {
  if (value === null || typeof value !== 'object' || seen.has(value)) return;
  seen.add(value);
  for (const item of Object.values(value)) freeze(item, seen);
  Object.freeze(value);
}

export function readonlySnapshot<T>(value: T): DeepReadonly<T> {
  const result = cloneValue(value);
  freeze(result, new Set());
  return result as DeepReadonly<T>;
}
