import { readable, writable } from 'svelte/store'
import type { Readable } from 'svelte/store'
import { socket } from './socket'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader'
import bunnyOBJ from '../assets/bunny.obj?raw'

const objLoader = new OBJLoader()

type Mesh = any

async function objStringToMesh(objString:string):Promise<Mesh> {
    return new Promise(resolve => {
        const blob = new Blob([objString], { type: 'text/plain' })
        const url = window.URL.createObjectURL(blob)
        objLoader.load(url, (data) => {
            resolve(data)
        })
    })
}



export const meshStore = writable<Mesh>(null, (set) => {
    // Start off with a default bunny for testing.
    objStringToMesh(bunnyOBJ).then((bunny) => {
        set(bunny)
    })
})

    
const losses: any = {};
export const lossStore = writable<Record<string, number[]>>({});

socket.addEventListener('message', async (event) => {
    const data = JSON.parse(event.data);
    for (const [name, loss] of Object.entries(data)) {
        const history = losses[name] = losses[name] ?? Array(100).fill(loss);
        history.push(loss);
        if (history.length > 100) {
            history.shift();
        } 
    }

    lossStore.set(losses);
    meshStore.set(await objStringToMesh(data.obj));
})
