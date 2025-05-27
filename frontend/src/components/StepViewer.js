import React, { useRef, useEffect, useState, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { 
  OrbitControls, 
  PerspectiveCamera, 
  Grid, 
  AdaptiveDpr, 
  AdaptiveEvents,
  BakeShadows,
  Stats
} from '@react-three/drei';
import * as THREE from 'three';
import initOpenCascade from 'opencascade.js';

function Model({ stepData }) {
  const [geometry, setGeometry] = useState(null);
  const [error, setError] = useState(null);
  const meshRef = useRef();

  useEffect(() => {
    async function loadStep() {
      try {
        if (!stepData) {
          console.warn('No STEP data provided');
          return;
        }

        console.log('Initializing OpenCascade.js...');
        const oc = await initOpenCascade();
        
        // Write STEP data to virtual filesystem
        console.log('Writing STEP data to virtual filesystem...');
        const uint8Array = new TextEncoder().encode(stepData);
        oc.FS.writeFile('model.step', uint8Array);
        
        // Create and initialize STEP reader
        console.log('Reading STEP file...');
        const stepReader = new oc.STEPControl_Reader_1();
        const readResult = stepReader.ReadFile('model.step');
        console.log('Read result:', readResult);
        
        if (readResult === oc.IFSelect_RetDone) {
          console.log('Transferring roots...');
          const transferResult = stepReader.TransferRoots(new oc.Message_ProgressRange_1());
          console.log('Transfer result:', transferResult);
          
          console.log('Getting shape...');
          const shape = stepReader.OneShape();
          
          if (shape.IsNull()) {
            throw new Error('No valid shape found in STEP file');
          }
          
          // Create a triangulation of the shape
          console.log('Creating triangulation...');
          new oc.BRepMesh_IncrementalMesh_2(shape, 0.1, false, 0.1, false);
          
          // Extract geometry data
          console.log('Extracting geometry...');
          const explorer = new oc.TopExp_Explorer_2(shape, oc.TopAbs_ShapeEnum.TopAbs_FACE, oc.TopAbs_ShapeEnum.TopAbs_SHAPE);
          const vertices = [];
          const indices = [];
          let indexOffset = 0;
          let faceCount = 0;
          
          while (explorer.More()) {
            faceCount++;
            const face = oc.TopoDS.Face_1(explorer.Current());
            const location = new oc.TopLoc_Location_1();
            const triangulation = oc.BRep_Tool.Triangulation(face, location).get();
            
            if (!triangulation.IsNull()) {
              // Get vertices
              const nodes = triangulation.Nodes();
              const nodeCount = nodes.Length();
              console.log(`Processing face ${faceCount} with ${nodeCount} nodes...`);
              
              for (let i = 1; i <= nodeCount; i++) {
                const point = nodes.Value(i);
                vertices.push(point.X(), point.Y(), point.Z());
              }
              
              // Get triangles
              const triangles = triangulation.Triangles();
              const triangleCount = triangles.Length();
              console.log(`Face ${faceCount} has ${triangleCount} triangles`);
              
              for (let i = 1; i <= triangleCount; i++) {
                const triangle = triangles.Value(i);
                indices.push(
                  triangle.Value(1) - 1 + indexOffset,
                  triangle.Value(2) - 1 + indexOffset,
                  triangle.Value(3) - 1 + indexOffset
                );
              }
              
              indexOffset += nodeCount;
            }
            
            explorer.Next();
          }
          
          console.log(`Processed ${faceCount} faces`);
          console.log('Total vertices:', vertices.length / 3);
          console.log('Total triangles:', indices.length / 3);
          
          if (vertices.length === 0 || indices.length === 0) {
            throw new Error('No geometry data extracted from STEP file');
          }
          
          // Create Three.js geometry
          console.log('Creating Three.js geometry...');
          const newGeometry = new THREE.BufferGeometry();
          newGeometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
          newGeometry.setIndex(indices);
          newGeometry.computeVertexNormals();
          
          // Center and scale geometry
          newGeometry.computeBoundingBox();
          const bbox = newGeometry.boundingBox;
          console.log('Bounding box:', bbox);
          
          newGeometry.center();
          const maxDim = Math.max(
            bbox.max.x - bbox.min.x,
            bbox.max.y - bbox.min.y,
            bbox.max.z - bbox.min.z
          );
          const scale = 5 / maxDim;
          newGeometry.scale(scale, scale, scale);
          
          console.log('Geometry created successfully');
          setGeometry(newGeometry);
          setError(null);
          
          // Cleanup
          oc.FS.unlink('model.step');
        } else {
          throw new Error('Failed to read STEP file');
        }
      } catch (error) {
        console.error('Error processing STEP data:', error);
        setError(error.message);
      }
    }
    
    loadStep();
  }, [stepData]);

  if (error) {
    console.error('Rendering error:', error);
    return null;
  }

  if (!geometry) {
    console.log('No geometry available yet');
    return null;
  }

  return (
    <group>
      {/* Debug axes helper */}
      <axesHelper args={[5]} />
      
      {/* Main mesh with enhanced material */}
      <mesh 
        ref={meshRef} 
        castShadow 
        receiveShadow
        renderOrder={1}
      >
        <bufferGeometry {...geometry} />
        <meshStandardMaterial 
          color="#67a3d9"
          roughness={0.4}
          metalness={0.6}
          side={THREE.DoubleSide}
          flatShading={false}
        />
      </mesh>
      
      {/* Enhanced edges */}
      <lineSegments renderOrder={2}>
        <edgesGeometry args={[geometry]}>
          <lineBasicMaterial 
            color="#000000" 
            transparent={true} 
            opacity={0.2}
            depthTest={true}
            depthWrite={false}
            linewidth={1}
          />
        </edgesGeometry>
      </lineSegments>
    </group>
  );
}

function StepViewer({ stepData }) {
  const showStats = process.env.NODE_ENV === 'development';

  return (
    <div style={{ width: '100%', height: '600px', background: '#f0f0f0' }}>
      <Canvas 
        shadows="soft" 
        dpr={[1, 2]} 
        camera={{ position: [15, 15, 15], fov: 50, near: 1, far: 1000 }}
        gl={{ 
          antialias: true,
          alpha: false,
          preserveDrawingBuffer: true,
          logarithmicDepthBuffer: true,
          powerPreference: "high-performance",
          stencil: false
        }}
      >
        <Suspense fallback={null}>
          {/* Performance optimizations */}
          <AdaptiveDpr pixelated />
          <AdaptiveEvents />
          <BakeShadows />
          {showStats && <Stats />}

          {/* Enhanced lighting setup */}
          <ambientLight intensity={0.5} />
          <hemisphereLight 
            intensity={1} 
            color="#ffffff" 
            groundColor="#444444" 
            position={[0, 50, 0]} 
          />
          <directionalLight
            position={[10, 10, 10]}
            intensity={0.5}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
          />
          <directionalLight
            position={[-10, -10, -10]}
            intensity={0.2}
          />

          {/* Model */}
          <Model stepData={stepData} />

          {/* Enhanced camera and controls */}
          <PerspectiveCamera makeDefault position={[15, 15, 15]} />
          <OrbitControls 
            makeDefault
            enableDamping
            dampingFactor={0.05}
            rotateSpeed={0.8}
            minDistance={5}
            maxDistance={100}
            enablePan={true}
            panSpeed={0.8}
            minPolarAngle={0}
            maxPolarAngle={Math.PI / 1.5}
            target={[0, 0, 0]}
          />

          {/* Enhanced grid with better visibility */}
          <Grid
            position={[0, -2, 0]}
            args={[30, 30]}
            cellSize={1}
            cellThickness={0.5}
            cellColor="#6f6f6f"
            sectionSize={5}
            sectionThickness={1}
            sectionColor="#9d4b4b"
            fadeDistance={30}
            fadeStrength={1}
            followCamera={false}
            infiniteGrid={true}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}

export default StepViewer; 