# DisasterView — Annotation Guide

This guide defines each of the 10 semantic segmentation classes in DisasterView,
including precise definitions, edge cases, and disambiguation rules for UAV/drone
aerial imagery of natural disasters.

---

## Annotation Method

DisasterView annotations were generated automatically using a CLIP + k-means pipeline:

1. **CLIP class scoring**: Each frame is scored against 10 text prompts using CLIP
   (ViT-L/14). The prompt with the highest cosine similarity identifies the dominant class.
2. **k-means segmentation**: k-means clustering (k=10) on LAB color space segments the
   image into regions. Regions are assigned to the dominant class or background based
   on spatial priors.
3. **CLIP verification**: Each annotated frame is re-scored; frames with confidence < 0.22
   are flagged in `quality_report.csv`.

Annotations are in **YOLO segmentation format** (one .txt per frame):
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
Coordinates are normalized to [0, 1] relative to image width/height.

---

## Class Definitions

### 0 — background

**Definition:** Any region that is not one of the 9 named disaster-relevant classes.
Includes clear sky, undamaged open ground, ocean/sea (when no flooding event), and regions
outside the primary disaster zone.

**From UAV perspective:** Large uniform regions with no structural or hazard features.
Sky patches at frame edges, intact grass fields far from damage.

**Edge cases:**
- Smoke-free sky in a wildfire frame → `background`
- Intact farmland adjacent to a flood zone → `background` (not `vegetation`)
- Shadow cast on ground → `background`

---

### 1 — building_damaged

**Definition:** Structures that have suffered visible physical damage: partial or total collapse,
exposed reinforcement, missing roofs, fire damage, flood damage to the structure.

**From UAV perspective:** Irregular rooflines, collapsed sections visible from above, debris
adjacent to structure, discolored or missing roof panels.

**Edge cases:**
- Partially flooded intact building → annotate the structural damage component as `building_damaged`
- Building with smoke rising from it → `building_damaged` (structure) + `fire_smoke` (smoke plume)
- Rubble pile that was formerly a building → `debris_rubble` (once structural form is lost)

---

### 2 — building_intact

**Definition:** Structures with no visible damage. Regular roofline, intact walls, normal
appearance from aerial view.

**From UAV perspective:** Regular grid patterns of roofs, consistent coloring, no debris
immediately adjacent to or on the structure.

**Edge cases:**
- Building in a flood zone but not visibly damaged → `building_intact`
- Building partially obscured by smoke → `building_intact` if damage is not visible
- Pre-existing older/weathered building → `building_intact` unless collapse is visible

---

### 3 — debris_rubble

**Definition:** Piles or fields of broken material: concrete chunks, wood fragments, displaced
soil, mixed destruction remnants. Distinct from a building in that the original structural
form is no longer discernible.

**From UAV perspective:** Irregular texture, mixed gray/brown/white patches, no regular structure.
Often adjacent to or overlapping `building_damaged` zones.

**Edge cases:**
- Small debris scattered on a road → `road_blocked` (road takes precedence if the road is the
  primary feature) or `debris_rubble` (if debris is the primary mass)
- Earthquake landslide material → `debris_rubble`
- Sand/sediment deposited by flooding → `water_flood` if still wet; `debris_rubble` if dried sediment

---

### 4 — fire_smoke

**Definition:** Active flame, fire glow, or smoke plumes (white, gray, or black). Applies to
wildfires, structural fires post-earthquake, and vehicle fires.

**From UAV perspective:** Bright orange/red regions (fire), diffuse white/gray masses (smoke),
or dark billowing columns (black smoke).

**Edge cases:**
- Dust cloud from building collapse → `debris_rubble` (not `fire_smoke`)
- Steam from a fire-suppression operation → `fire_smoke`
- Burn scars (already-burned ground) → `background` (no active fire/smoke)

---

### 5 — road_blocked

**Definition:** Road or pathway that is obstructed and impassable due to debris, flooding,
landslide, damaged pavement, or collapsed bridge.

**From UAV perspective:** Linear infrastructure (road) covered or broken, with visible obstruction.

**Edge cases:**
- Flooded road → `road_blocked` (obstruction takes precedence over road class)
- Road with minor debris on shoulder → `road_clear` if the carriageway remains passable
- Destroyed bridge → `road_blocked`

---

### 6 — road_clear

**Definition:** Road or pathway that remains intact and passable, showing no significant obstruction.

**From UAV perspective:** Clear linear markings, smooth surface, no debris on carriageway.

**Edge cases:**
- Road with parked vehicles on shoulder → `road_clear` (vehicles do not block traffic flow)
- Dirt track in rural area → `road_clear` if passable
- Road adjacent to flood zone but not flooded → `road_clear`

---

### 7 — vegetation

**Definition:** Trees, forest canopy, grass, crops, or any significant plant cover visible
from above.

**From UAV perspective:** Green or brown (dry season) organic texture patterns. Forest canopy
appears as dense irregular green masses; grass as uniform green.

**Edge cases:**
- Burned vegetation → `background` (no living plant cover) or `debris_rubble` (if fallen trees)
- Flooded vegetation (still visible above water) → `vegetation`
- Isolated trees in an urban area → `vegetation` if they form a meaningful region

---

### 8 — vehicle

**Definition:** Individual vehicles: cars, trucks, emergency vehicles, military vehicles.
Does not include aircraft or watercraft.

**From UAV perspective:** Small rectangular/rounded objects on roads or parking areas, typically
1–3 meters in size at common UAV altitudes. May be clustered in parking areas.

**Edge cases:**
- Emergency vehicle at disaster scene → `vehicle`
- Vehicle partially submerged in flood → `vehicle` (if still identifiable as a vehicle)
- Vehicle crushed under rubble → `debris_rubble` (if structural form lost) or `vehicle` (if identifiable)
- Very high altitude where vehicles are sub-pixel → skip (no annotation for sub-pixel objects)

---

### 9 — water_flood

**Definition:** Standing floodwater, inundated areas, water flowing through normally dry areas.
Applies specifically to flooding events, not to natural water bodies unless they have expanded
due to flooding.

**From UAV perspective:** Reflective, dark or murky flat surfaces in areas not normally covered
by water. Often brown/gray from sediment.

**Edge cases:**
- Normal river within its banks → `background` (not flood water)
- River overflowing its banks → `water_flood` for the overflow region only
- Pool of water after firefighting → `water_flood` only if significant
- Puddles from rain (not disaster-related) → `background`

---

## Quality Control Process

1. **Automated CLIP confidence scoring**: Every annotated frame receives a per-class confidence
   score. Frames below 0.22 are flagged in `quality_report.csv` (`flagged=True`).
2. **Flagged frame review**: Flagged frames are preserved in the dataset but identified so
   researchers can choose to exclude them in training.
3. **No manual correction**: All annotations are machine-generated. Manual review was not
   performed for this release.

For downstream use, filtering to `flagged=False` in `quality_report.csv` gives the
27,280 highest-confidence frames (91.3% of total).

---

## Known Annotation Limitations

- **Dominant-class bias**: The CLIP assignment identifies the single most prominent class
  per frame. Frames with multiple equally prominent classes may have incorrect dominant
  class assignments.
- **k-means color-based segmentation**: Color-based clustering may conflate visually similar
  classes (e.g., gray debris and gray road, brown flood water and brown bare earth).
- **Small object under-annotation**: Objects smaller than ~50×50 pixels (vehicles at high
  altitude) may be missed.
- **Class boundary ambiguity**: Boundaries between `building_damaged` and `debris_rubble`
  are approximate; the transition from damaged structure to pure rubble is continuous.
