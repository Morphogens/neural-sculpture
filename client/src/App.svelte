<script lang="ts">
import { onMount } from 'svelte'
import * as THREE from 'three'
import { debounce } from 'ts-debounce'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'; 
import { socket, socketOpen } from './stores/socket'
import { mesh } from './stores/mesh'

socketOpen.subscribe(console.log)

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const lastCamera = new THREE.Vector3()

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
const controls = new OrbitControls( camera, renderer.domElement )

// set size
renderer.setSize(window.innerWidth, window.innerHeight);

// add canvas to dom
document.body.appendChild(renderer.domElement);

// add axis to the scene
const axis = new THREE.AxesHelper(10);
scene.add(axis);


const ambientLight = new THREE.AmbientLight( 0xcccccc, 0.4 );
scene.add( ambientLight );

const directionalLight = new THREE.DirectionalLight( 0xffffff, 0.8 );
directionalLight.position.set( -100, -100, -100 )
scene.add( directionalLight );

camera.position.x = 100;
camera.position.y = 100;
camera.position.z = 100;

camera.lookAt(scene.position);

let lastMesh = null
mesh.subscribe($mesh => {
    if (lastMesh) {
        scene.remove(lastMesh)
    }
    if ($mesh) {
        console.log('Got new mesh')
        scene.add($mesh)
        lastMesh = $mesh
    }
})

function loop(): void {
    requestAnimationFrame(loop);
    update();
    renderer.render(scene, camera);
}

const sendNewCamera = debounce(() => {
    console.log('new camera')
    socket.send(JSON.stringify({
        message: 'camera',
        data: camera.position.toArray()
    }))
}, 200)

function update(): void {
    // raycast()
    if ($socketOpen && !camera.position.equals(lastCamera)) {
        sendNewCamera()
        lastCamera.copy(camera.position)
    }
    controls.update()
}

function raycast() {
    // update the picking ray with the camera and mouse position
	raycaster.setFromCamera( mouse, camera );
    // calculate objects intersecting the picking ray
    const intersects = raycaster.intersectObjects( scene.children );
    console.log(intersects.length);
    for ( let i = 0; i < intersects.length; i ++ ) {
        intersects[ i ].object.material.color.set( 0xff0000 );
    }
}

// function onMouseMove( event ) {
//     // calculate mouse position in normalized device coordinates
//     // (-1 to +1) for both components
//     // mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
//     // mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
//     // console.log(mouse);
// }
// document.onmousemove = onMouseMove

loop();

let clip_input:HTMLInputElement

onMount(() => {
    clip_input.onchange = () => {
        console.log('New text value', clip_input.value)
        socket.send(JSON.stringify({
            message: 'prompt',
            data: clip_input.value
        }))
    }
})

</script>

<div id='container'>
    <input type="text" value="Bunny" bind:this={clip_input}>
</div>

<style>
    :global(body) {
        margin: 0px;
        padding: 0px;
    }
    #container {
        width: 200px;
        position: absolute;
    }
        
    input[type=text] {
        /* width: 100%; */
        padding: 12px 20px;
        margin: 8px 0;
        box-sizing: border-box;
    }
</style>
