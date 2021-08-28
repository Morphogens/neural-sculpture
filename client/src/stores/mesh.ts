import { readable } from 'svelte/store'
import type { Readable } from 'svelte/store'
import { socket } from './socket'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader'
const loader = new OBJLoader()

type Mesh = any
export const mesh:Readable<null | Mesh> = readable(null, (set) => {    
    socket.addEventListener('message', function (event) {
        console.log('Message from server ', event)
        const objString = event.data
        const blob = new Blob([objString], {type: 'text/plain'});
        const url = window.URL.createObjectURL(blob)
        loader.load(url, (data) => {
            set(data)
        })
    })
})
