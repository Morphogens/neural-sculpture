<script lang="ts">
import { onMount } from 'svelte'
import * as THREE from 'three'
import { debounce } from 'ts-debounce'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'; 
import { socket, socketOpen } from './stores/socket'
import { mesh } from './stores/mesh'

////////////////////////////////////////////////////////////////////////////////
let clip_input:HTMLInputElement
let mouseDown = false
let shiftDown = false
const mouse = new THREE.Vector2();
const lastCamera = new THREE.Vector3()

////////////////////////////////////////////////////////////////////////////////
// Setup 3d seen
const raycaster = new THREE.Raycaster();
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

// const ambientLight = new THREE.AmbientLight( 0xcccccc, 0.4 );
// scene.add( ambientLight );

// const directionalLight = new THREE.DirectionalLight( 0xffffff, 0.8 );
// directionalLight.position.set( -100, -100, -100 )
// scene.add( directionalLight );

const lights = [];
lights[ 0 ] = new THREE.PointLight( 0xffffff, 1, 0 );
lights[ 1 ] = new THREE.PointLight( 0xffffff, 1, 0 );
lights[ 2 ] = new THREE.PointLight( 0xffffff, 1, 0 );

lights[ 0 ].position.set( 0, 200, 0 );
lights[ 1 ].position.set( 100, 200, 100 );
lights[ 2 ].position.set( - 100, - 200, - 100 );

scene.add( lights[ 0 ] );
scene.add( lights[ 1 ] );
scene.add( lights[ 2 ] );

camera.position.x = 100;
camera.position.y = 100;
camera.position.z = 100;

camera.lookAt(scene.position);

const geometry = new THREE.SphereGeometry( 2.0, 8, 8 );
// const material = new THREE.MeshBasicMaterial( { color: 0xffff00 } );
const meshMaterial = new THREE.MeshPhongMaterial({
    color: 0x156289,
    emissive: 0x072534,
    side: THREE.DoubleSide,
    flatShading: true
})

const sphere = new THREE.Mesh( geometry, meshMaterial );
scene.add( sphere )
////////////////////////////////////////////////////////////////////////////////

function messageServer(message:string, data:any) {
    if ($socketOpen) {
        socket.send(JSON.stringify({ message, data }))
    }
}

// Listen for new meshes from the server.
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

const sendNewCamera = debounce(() => {
    messageServer('camera', camera.position.toArray())
}, 200)

function loop(): void {
    requestAnimationFrame(loop);
    update();
    renderer.render(scene, camera);
}

function update(): void {
    raycast()
    if ($socketOpen && !camera.position.equals(lastCamera)) {
        sendNewCamera()
        lastCamera.copy(camera.position)
    }
    controls.update()
}

////////////////////////////////////////////////////////////////////////////////
// Detect which position is being moused over.
let mouseOverMesh = false
function raycast() {
    if ($mesh) {
        raycaster.setFromCamera( mouse, camera );
        const intersects = raycaster.intersectObjects($mesh.children);
        if (intersects.length) {
            mouseOverMesh = true
            const { point } = intersects[0]
            sphere.position.copy(point)
            document.body.style.cursor = 'none'
            if (mouseDown){
                messageServer('cursor', {
                    mouseDown,
                    additive: !shiftDown,
                    point: point.toArray()
                })
            }
        } else {
            mouseOverMesh = false
            document.body.style.cursor = 'default'
            messageServer('cursor', null)
        }
    }
}
function onMouseMove(event) {
    // calculate mouse position in normalized device coordinates
    // (-1 to +1) for both components
    mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
    mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
}
const sphereColors = {
    'inactive': {r: 0.2, g: 0.2, b: 0.2},
    'positive': {r: 0.1, g: 1.0, b: 0.5372549019607843},
    'negative': {r: 0.9, g: 0.3843137254901961, b: 0.5372549019607843},
}
$: sphere.visible = mouseOverMesh
$: sphereColorName = mouseDown ? (shiftDown ? 'negative' : 'positive') : 'inactive'
$: sphere.material.color= sphereColors[sphereColorName]
////////////////////////////////////////////////////////////////////////////////

onMount(() => {
    clip_input.onchange = () => {
        console.log('New text value', clip_input.value)
        messageServer('prompt', clip_input.value)
    }
    loop()
})

</script>

<svelte:body
    on:mousemove={onMouseMove}
    on:mousedown={() => mouseDown=true}
    on:mouseup={() => mouseDown=false}
    on:keydown={(event) => shiftDown = event.shiftKey}
    on:keyup={(event) => shiftDown = event.shiftKey}
/>
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
