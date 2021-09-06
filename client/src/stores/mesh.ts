import { readable } from 'svelte/store'
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

export const mesh:Readable<null | Mesh> = readable(null, (set) => {
    // Start off with a default bunny for testing.
    // objStringToMesh(bunnyOBJ).then((bunny) => {
    //     set(bunny)
    // })
    
    socket.addEventListener('message', async (event) => {
        const objString = event.data
        set(await objStringToMesh(objString))
    })
})
