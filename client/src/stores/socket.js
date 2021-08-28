import { writable, readable } from 'svelte/store'

// export const socket = new WebSocket('ws://192.168.193.42:9999')
export const socket = new WebSocket('ws://localhost:9999/ws')

export const socketOpen = readable(false, (set) => {    
    socket.addEventListener('open', (event) => {
        console.log('SOCKET CONNECTED');
        set(true)
    })    
    socket.addEventListener('close', (event) => {
        console.log('SOCKET DISCONNECTED');
        set(false)
    })
})
