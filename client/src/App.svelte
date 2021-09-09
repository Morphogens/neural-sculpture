<script lang="ts">
import { onMount } from 'svelte'
import * as THREE from 'three'
import { debounce } from 'ts-debounce'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'; 
import { socket, socketOpen } from './stores/socket'
import { lossStore, meshStore as mesh } from './stores/mesh'
import LossHistory from "./LossHistory.svelte";
import * as knobby from 'svelte-knobby';

// sent to server as sculp_settings
const sculpControls = knobby.panel({
    sculp_enabled: false,
    prompt: "A sculpture of a bunny rabbit",
    learning_rate: 1,
    batch_size: 1,
    current_coord: "none",
    optimization_region: {
        $label: 'optimization_region',
        value: 6,
        min: 2,
        max: 64,
        step: 2
    },
    grid_resolution: {
        $label: 'grid_resolution',
        value: 64,
        min: 4,
        max: 64,
        step: 4
    },
    reset_mesh: () => {messageServer("initialize", "cat.npy")},
});


////////////////////////////////////////////////////////////////////////////////
let mouseDown = false
let mouseClicked = false
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
    if (lastMesh == null){
        messageServer("initialize", "cat.npy")
    }
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
            if (mouseClicked){
                messageServer('coord', {
                    mouseClicked,
                    additive: !shiftDown,
                    point: point.toArray()
                })
                console.log("COORD INFO SENT TO THE SERVER")
                console.log("COORD ", point.toArray())
                mouseClicked = false
                
                const intCoord = point.toArray().reduce((total, x) => {return total + parseInt(x).toString() + ", "}, "")
                $sculpControls.current_coord = intCoord
            }

        } else {
            mouseOverMesh = false
            document.body.style.cursor = 'default'
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

const submitSettings = debounce((settings:any) => {
    messageServer("sculp_settings", settings)
}, 100);

$: submitSettings($sculpControls);

onMount(() => {
    loop()
})

function resetClicked() {
    messageServer("initialize", "12140_Skull_v3_L2.npy")
}

let isSculping = false;
function onKeyDown(e: KeyboardEvent) {
    shiftDown = e.shiftKey;
    if (e.key.toLowerCase() === "t") {
        console.log("SCUlPTING!")
        // if (!isSculping){

        //     isSculping = true;
        //     messageServer("sculp_mode", {is_sculping: isSculping})
        // }
        $sculpControls.sculp_enabled = !$sculpControls.sculp_enabled
    };

}

function onKeyUp(e: KeyboardEvent) {
    shiftDown = e.shiftKey;
    // if (e.key.toLowerCase() === "t") {
    //     console.log("SCUlPTING STOPPED!")
        
    //     isSculping = false;
    //     messageServer("stop_sculp_mode", {is_sculping: isSculping})
    // };

}

</script>

<svelte:body
    on:mousemove={onMouseMove}
    on:mousedown={() => mouseDown=true}
    on:click={() => mouseClicked=true}
    on:mouseup={() => mouseDown=false}
    on:keydown={onKeyDown}
    on:keyup={onKeyUp}
/>
<div id='container'>
    <div class="instructions">
        <div>instructions</div>
        <p>Press "T" to toggle sculping mode on/off. Click an area to target the sculping</p>
    </div>
    <div class="connection">
        <div class="indicator" class:live={$socketOpen} class:closed={!$socketOpen}/>Connection: {$socketOpen ? "Live" : "Disconnected" }
    </div>
    <button on:click={resetClicked}>
        <div>Reset Mesh + Optimizer</div>
         <div>(takes a few seconds)</div>
    </button>
    <LossHistory points={$lossStore["camera"]}/>
</div>
{$sculpControls}

<style>
    .instructions {
        background: white;
        padding: 4px;
        margin: 1px;
    }
    .connection {
        background: white;
        display: flex;
        align-items: center;
        padding: 4px;
    }

    .sculp-mode {
        background: white;
        display: flex;
        align-items: center;
        padding: 4px;
    }
    .indicator {
        width: 10px;
        height: 10px;
        border-radius: 100%;
        margin-right: 2px;
    }
    .indicator.live {
        background: #059669;
    }
    .indicator.closed {
        background: #DC2626
;
    }

    :global(body) {
        margin: 0px;
        padding: 0px;
    }
    #container {
        width: 200px;
        position: absolute;
        display: flex;
        flex-flow: column;
    }
        
    input[type=text] {
        /* width: 100%; */
        padding: 12px 20px;
        margin: 8px 0;
        box-sizing: border-box;
    }
</style>
