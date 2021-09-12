import { writable } from "svelte/store"
export type Mode = 'setting-sphere' | 'sculpting' | 'default'
export const mode = writable('default' as Mode)