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

function Model({ geometryData }) {
  const [geometry, setGeometry] = useState(null);
  const [error, setError] = useState(null);
  const meshRef = useRef();

  useEffect(() => {
    try {
      if (!geometryData || !geometryData.vertices || !geometryData.faces) {
        console.warn('No geometry data provided');
        return;
      }

      console.log('Creating Three.js geometry...');
      const newGeometry = new THREE.BufferGeometry();
      
      // Add vertices
      const vertices = new Float32Array(geometryData.vertices.flat());
      newGeometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
      
      // Add faces
      const indices = new Uint32Array(geometryData.faces.flat());
      newGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
      
      // Compute normals
      newGeometry.computeVertexNormals();
      
      // Center and scale geometry
      newGeometry.computeBoundingBox();
      const bbox = newGeometry.boundingBox;
      
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
      
    } catch (error) {
      console.error('Error creating geometry:', error);
      setError(error.message);
    }
  }, [geometryData]);

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
      
      {/* Wireframe mesh */}
      <lineSegments>
        <wireframeGeometry args={[geometry]} />
        <lineBasicMaterial 
          color="#00ff00"
          linewidth={1}
        />
      </lineSegments>
    </group>
  );
}

function StepViewer({ geometryData }) {
  const showStats = process.env.NODE_ENV === 'development';

  if (!geometryData) {
    return (
      <div style={{ width: '100%', height: '600px', background: '#2a2a2a', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ color: '#ffffff' }}>Loading geometry data...</p>
      </div>
    );
  }

  return (
    <div style={{ width: '100%', height: '600px', background: '#2a2a2a' }}>
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
        <color attach="background" args={['#2a2a2a']} />
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
          <Model geometryData={geometryData} />

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

          <Grid
            position={[0, -2, 0]}
            args={[30, 30]}
            cellSize={1}
            cellThickness={0.5}
            cellColor="#404040"
            sectionSize={5}
            sectionThickness={1}
            sectionColor="#707070"
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