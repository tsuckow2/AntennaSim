/**
 * ModelIRPreview — Three.js scene that renders a ModelIR as a 3D preview.
 *
 * Coordinate mapping: IR uses Z-up (physics convention); Three.js uses Y-up.
 *   Three.js [x, y, z] ← IR [x, z, -y]
 *
 * Entities rendered:
 *   box        → translucent BoxGeometry coloured by material
 *   sheetRect  → thin PlaneGeometry oriented to the IR normal
 *   lumpedPort → small sphere at the midpoint between terminals
 *   airbox/pml → wireframe bounding box
 */

import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Quaternion, Vector3 } from "three";
import type { ModelIR, Material, Vec3 } from "../../dsl/model-ir";
import type { PatternData } from "../../api/nec";
import { RadiationPattern3D } from "../three/RadiationPattern3D";
import { useUIStore } from "../../stores/uiStore";

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

function irToThree(v: Vec3): Vec3 {
  return [v[0], v[2], -v[1]];
}

function irSizeToThree(s: Vec3): Vec3 {
  return [s[0], s[2], s[1]]; // sizes are always positive — no sign flip needed
}

// ---------------------------------------------------------------------------
// Material colours
// ---------------------------------------------------------------------------

const COPPER_COLOR = "#B87333";
const CONDUCTOR_COLORS: Record<string, string> = {
  copper: COPPER_COLOR,
  gold: "#FFD700",
  silver: "#C0C0C0",
  aluminum: "#A8A9AD",
};
const DIELECTRIC_PALETTE = ["#4A90D9", "#7BC47E", "#E8A838", "#C46EB0", "#46C6C6"];

function getMaterialStyle(mat: Material): { color: string; opacity: number; wireframe: boolean } {
  const lower = mat.name.toLowerCase();

  // Conductors (very high conductivity)
  if (mat.conductivity_s_m !== undefined && mat.conductivity_s_m > 1e4) {
    for (const [k, c] of Object.entries(CONDUCTOR_COLORS)) {
      if (lower.includes(k)) return { color: c, opacity: 0.92, wireframe: false };
    }
    return { color: COPPER_COLOR, opacity: 0.92, wireframe: false };
  }

  // Near-vacuum / air
  if (mat.epsilon_r < 1.05 && !mat.conductivity_s_m) {
    return { color: "#8888AA", opacity: 0.06, wireframe: false };
  }

  // Dielectrics — cycle palette by name hash
  let hash = 0;
  for (const c of mat.name) hash = (hash * 31 + c.charCodeAt(0)) & 0xffffff;
  const color = DIELECTRIC_PALETTE[hash % DIELECTRIC_PALETTE.length]!;
  return { color, opacity: 0.45, wireframe: false };
}

// ---------------------------------------------------------------------------
// Scene geometry components
// ---------------------------------------------------------------------------

function MaterialsMap({ ir }: { ir: ModelIR }) {
  const styles = useMemo(
    () => new Map(ir.materials.map((m) => [m.name, getMaterialStyle(m)])),
    [ir.materials],
  );

  return (
    <>
      {ir.entities.map((entity) => {
        const style = styles.get(entity.material) ?? { color: "#888888", opacity: 0.5, wireframe: false };
        const [r, g, b] = irToThree(entity.kind === "box" ? entity.center : entity.center);
        void r; void g; void b; // used in render below

        if (entity.kind === "box") {
          const [px, py, pz] = irToThree(entity.center);
          const [sx, sy, sz] = irSizeToThree(entity.size as Vec3);
          return (
            <mesh key={entity.id} position={[px, py, pz]}>
              <boxGeometry args={[sx, sy, sz]} />
              <meshStandardMaterial
                color={style.color}
                transparent
                opacity={style.opacity}
                depthWrite={false}
              />
            </mesh>
          );
        }

        if (entity.kind === "sheetRect") {
          const [px, py, pz] = irToThree(entity.center);
          const [w, h] = entity.size;
          // IR normal → Three.js normal
          const irN = entity.normal as Vec3;
          const threeNormal = new Vector3(irN[0], irN[2], -irN[1]).normalize();
          const defaultNormal = new Vector3(0, 0, 1);
          const quat = new Quaternion().setFromUnitVectors(defaultNormal, threeNormal);

          return (
            <mesh
              key={entity.id}
              position={[px, py, pz]}
              quaternion={[quat.x, quat.y, quat.z, quat.w]}
            >
              <planeGeometry args={[w, h]} />
              <meshStandardMaterial
                color={style.color}
                transparent
                opacity={style.opacity}
                side={2 /* DoubleSide */}
                depthWrite={false}
              />
            </mesh>
          );
        }

        return null;
      })}

      {/* Lumped ports — separate positive and negative terminal markers */}
      {ir.ports.map((port) => {
        const [positiveX, positiveY, positiveZ] = irToThree(port.positive);
        const [negativeX, negativeY, negativeZ] = irToThree(port.negative);
        return (
          <group key={port.id}>
            <mesh position={[positiveX, positiveY, positiveZ]}>
              <sphereGeometry args={[0.001, 8, 8]} />
              <meshStandardMaterial color="#FF4444" emissive="#FF2222" emissiveIntensity={0.6} />
            </mesh>
            <mesh position={[negativeX, negativeY, negativeZ]}>
              <sphereGeometry args={[0.001, 8, 8]} />
              <meshStandardMaterial color="#4488FF" emissive="#2244FF" emissiveIntensity={0.6} />
            </mesh>
          </group>
        );
      })}

      {/* Airbox / PML — wireframe */}
      <AirboxWireframe ir={ir} />
    </>
  );
}

function AirboxWireframe({ ir }: { ir: ModelIR }) {
  const { center, size } = useMemo(() => {
    if (ir.boundaries.length === 0) return { center: null, size: null };

    // Compute bounding box of all geometry entities
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let hasGeom = false;

    for (const entity of ir.entities) {
      hasGeom = true;
      if (entity.kind === "box") {
        const [cx, cy, cz] = entity.center;
        const [sx, sy, sz] = entity.size;
        minX = Math.min(minX, cx - sx / 2); maxX = Math.max(maxX, cx + sx / 2);
        minY = Math.min(minY, cy - sy / 2); maxY = Math.max(maxY, cy + sy / 2);
        minZ = Math.min(minZ, cz - sz / 2); maxZ = Math.max(maxZ, cz + sz / 2);
      } else {
        const [cx, cy, cz] = entity.center;
        minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);
        minZ = Math.min(minZ, cz); maxZ = Math.max(maxZ, cz);
      }
    }

    if (!hasGeom) return { center: null, size: null };

    const padding = Math.max(...ir.boundaries.map((b) => b.padding_m));
    const irCenter: Vec3 = [
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2,
    ];
    const irSize: Vec3 = [
      maxX - minX + 2 * padding,
      maxY - minY + 2 * padding,
      maxZ - minZ + 2 * padding,
    ];

    return {
      center: irToThree(irCenter),
      size: irSizeToThree(irSize),
    };
  }, [ir]);

  if (!center || !size) return null;

  return (
    <mesh position={center}>
      <boxGeometry args={size} />
      <meshBasicMaterial color="#4466AA" wireframe transparent opacity={0.25} />
    </mesh>
  );
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

function EmptyState() {
  return (
    <mesh position={[0, 0, 0]}>
      <sphereGeometry args={[0.01, 8, 8]} />
      <meshBasicMaterial color="#444444" />
    </mesh>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface ModelIRPreviewProps {
  ir?: ModelIR;
  /** Optional far-field pattern to overlay on the antenna (NEC PatternData shape). */
  pattern?: PatternData | null;
}

export function ModelIRPreview({ ir, pattern }: ModelIRPreviewProps) {
  const theme = useUIStore((s) => s.theme);
  const bg = theme === "dark" ? "#0A0A0F" : "#E8E8ED";

  // Centroid and scale for the pattern overlay, from the IR entity bounds.
  const { center, scale } = useMemo(() => {
    if (!ir || ir.entities.length === 0) {
      return { center: [0, 0, 0] as [number, number, number], scale: 0.1 };
    }
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const entity of ir.entities) {
      const c = entity.center;
      // Expand by half-size to get bounds.
      let hx = 0, hy = 0, hz = 0;
      if (entity.kind === "box") {
        hx = entity.size[0] / 2; hy = entity.size[1] / 2; hz = entity.size[2] / 2;
      } else if (entity.kind === "sheetRect") {
        hx = entity.size[0] / 2; hy = entity.size[1] / 2; hz = 0;
      }
      minX = Math.min(minX, c[0] - hx); maxX = Math.max(maxX, c[0] + hx);
      minY = Math.min(minY, c[1] - hy); maxY = Math.max(maxY, c[1] + hy);
      minZ = Math.min(minZ, c[2] - hz); maxZ = Math.max(maxZ, c[2] + hz);
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;
    const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
    // IR (x,y,z) -> Three.js (x,z,-y).
    return {
      center: [cx, cz, -cy] as [number, number, number],
      scale: span * 0.6,
    };
  }, [ir]);

  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={{ position: [0.25, 0.2, 0.25], fov: 50, near: 0.0001, far: 200 }}
        style={{ background: bg }}
        gl={{ antialias: true }}
      >
        <ambientLight intensity={1.2} />
        <directionalLight position={[1, 2, 1]} intensity={1.5} />
        <directionalLight position={[-1, -1, -0.5]} intensity={0.4} />

        <OrbitControls makeDefault enableDamping dampingFactor={0.1} />

        {/* Axes hint — tiny lines at origin */}
        <axesHelper args={[0.05]} />

        {ir ? <MaterialsMap ir={ir} /> : <EmptyState />}

        {/* Far-field pattern overlay, centered on the antenna. */}
        {pattern && (
          <RadiationPattern3D
            pattern={pattern}
            scale={scale}
            opacity={0.55}
            center={center}
          />
        )}
      </Canvas>

      {!ir && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <p className="text-xs text-text-secondary bg-surface/80 rounded px-3 py-1.5">
            Run the script to see a 3D preview
          </p>
        </div>
      )}
    </div>
  );
}
