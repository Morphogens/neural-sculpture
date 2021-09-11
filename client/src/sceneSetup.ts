import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'; 

interface SceneData {
    scene: THREE.SCENE,
    raycaster: THREE.Raycaster,
    camera: THREE.PerspectiveCamera,
    controls: OrbitControls,
    sceneUpdate: Function
}

export default function(): SceneData {
    const raycaster = new THREE.Raycaster();
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer();
    const controls = new OrbitControls( camera, renderer.domElement )
    console.log(controls);
    
    // set size
    renderer.setSize(window.innerWidth, window.innerHeight);

    // add canvas to dom
    document.body.appendChild(renderer.domElement);

    // add axis to the scene
    const axis = new THREE.AxesHelper(10);
    scene.add(axis)

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

    const sceneUpdate = () => {
        controls.update()
        renderer.render(scene, camera)
    }

    return { scene, raycaster, camera, controls, sceneUpdate }
}