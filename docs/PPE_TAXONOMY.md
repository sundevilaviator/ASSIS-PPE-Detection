# ASSIS PPE Taxonomy and Conditional Compliance Model

**Status:** Design specification · Phase 1.5 planning
**Date:** 2026-08-13

This document defines the PPE classes ASSIS aims to detect, assesses what a
camera can and cannot verify for each, and specifies the conditional logic
that determines when a missing item constitutes an actual violation.

It exists because the Phase 1 class set (`gloves`, `helmet`, `vest`) was
inherited from a general-industry construction dataset and does not
correspond to ramp PPE requirements. `helmet` in particular is a taxonomy
error: ramp personnel wear caps and hearing protection, not rigid hard hats.

---

## 1. Ramp PPE classes

Ordered by consequence of absence, not by ease of detection.

| Class | Protects against | Regulatory basis (verify before citing) |
|---|---|---|
| High-visibility vest / apparel | Struck-by: aircraft, tugs, GSE. The leading fatal ramp hazard. | OSHA 29 CFR 1910.132; carrier ground ops manuals; IATA IGOM/AHM |
| Hearing protection | Permanent noise-induced hearing loss from engines and GSE | OSHA 29 CFR 1910.95 |
| Safety footwear | Rolling loads, dropped cargo, slick or icy surfaces | OSHA 29 CFR 1910.136 |
| Gloves | Abrasion, sharp cargo edges, hot and cold metal | OSHA 29 CFR 1910.138 |
| Eye protection | Dust, wind-borne debris, deicing fluid splash | OSHA 29 CFR 1910.133 |

> **Citation note.** Ramp worker PPE derives principally from OSHA general
> industry standards, carrier ground operations manuals, and IATA IGOM/AHM
> guidance. FAA 14 CFR Part 139 governs airport certification and does not
> itself prescribe worker PPE. Any regulatory citation in a filing or
> publication should be verified against the source text.

---

## 2. What a camera can and cannot verify

This is the most important section. Several PPE attributes are physically
invisible to an optical sensor. Claiming to verify them would be a claim the
sensor cannot support.

| Class | Detectable | **Cannot be verified optically** |
|---|---|---|
| Hi-vis vest | Presence, approximate coverage | Retroreflective performance class (ANSI/ISEA 107 Type/Class); whether garment is soiled past effectiveness |
| Hearing protection | Earmuffs, when visible in profile | **Inserted earplugs are invisible.** Non-detection must never be treated as non-compliance. Also cannot verify NRR rating or correct insertion. |
| Footwear | Closed-toe boot present | **Steel/composite toe cap, anti-skid sole rating, puncture resistance.** All internal or microscopic. |
| Gloves | Presence on hands, when not occluded | Material, cut rating, thermal rating |
| Eye protection | Glasses or goggles, at sufficient resolution | Impact rating (ANSI Z87.1), UV/splash suitability |

**Design consequence.** ASSIS reports *observable proxies*, not certified
compliance. The correct output phrasing is "closed-toe boot detected", never
"steel-toe compliance verified". This distinction must survive into the UI,
the documentation, and any publication.

---

## 3. Detectability at operational camera distance

Assessed against Phase 1 empirical results (2026-08-13, `models/best.pt`).

| Class | Difficulty | Basis |
|---|---|---|
| Hi-vis vest | Low | Validated: 0.883 and 0.920 on held-out aviation photographs |
| Hearing protection | Medium | Head-mounted equipment detected at 0.815 when inference resolution raised to 960px; currently mislabelled as `helmet` |
| Gloves | High | Zero detections on aviation imagery even at conf=0.01. Small, frequently occluded by cargo and tooling. |
| Footwear | High | Small, low contrast against pavement, occluded by equipment |
| Eye protection | Very high | At wide-shot apron distance, spans only a few pixels. May require close-range cameras (e.g. at stand entry) rather than general apron coverage. |

**Empirical finding.** Inference resolution is a first-order variable for
small classes. Raising `imgsz` from 640 to 960 moved head-mounted equipment
from undetected to 0.815 confidence on identical imagery, with no retraining.
Small-object classes should be trained and inferred at >=960px.

---

## 4. Conditional compliance model

**Core principle: the detector reports observations; a separate policy layer
decides whether an observation is a violation.** Requirement rules must not
be embedded in the model.

Rationale: PPE requirements are conditional on weather, task, and operational
state. A system that flags missing eye protection on a calm, dry, warm day
produces continuous false violations. Operator trust, once lost, is not
recovered — an unreliable alert is worse than an absent one.

### 4.1 Requirement conditions

| Class | Always required | Conditionally required |
|---|---|---|
| Hi-vis vest | Yes — all airside personnel, all conditions | — |
| Hearing protection | Effectively yes when engines or GSE operating | Relaxable only in confirmed low-noise zones |
| Footwear | Yes | Anti-skid particularly relevant in ice/precipitation |
| Gloves | No | Cold temperature; cargo/baggage handling; hot surfaces |
| Eye protection | No | **Deicing operations**; high wind or blowing dust; low visibility with debris |

### 4.2 Condition data sources

| Signal | Source | Confidence |
|---|---|---|
| Wind speed and gusts | METAR | High |
| Temperature, dewpoint | METAR | High |
| Precipitation type and intensity | METAR | High |
| Visibility | METAR | High |
| Day / night | Solar position calculation | High |
| Deicing in progress | Deicing vehicle detected in frame, or airport ops feed | **Medium — heuristic** |
| Engines running / jet blast | Ops data or audio; not reliably inferable from image alone | **Low** |

METAR is free, standardised, and published for every US towered airport. It
is aviation-native infrastructure and represents a concrete integration
advantage over general-industry PPE analytics products, which have no
equivalent environmental context source.

### 4.3 Documented assumptions and limits

These are limitations of the approach, recorded so they are not later
mistaken for solved problems:

1. **METAR is airport-wide.** It does not capture localised jet blast,
   propwash, or a gust at an individual stand. Stand-level conditions may
   differ materially from the reported field observation.
2. **METAR is periodic.** Routine observations are hourly, with SPECIs on
   significant change. Rapidly shifting conditions may lag.
3. **Deicing inference is a heuristic.** Detecting a deicing vehicle in frame
   suggests, but does not establish, that deicing is underway at that stand.
4. **Absence of detection is not absence of PPE.** Especially for earplugs
   (invisible) and gloves (occluded). Policy must distinguish
   "not detected" from "confirmed absent".

### 4.4 Output states

Three states, not two. Binary pass/fail cannot express detector uncertainty.

| State | Meaning |
|---|---|
| `COMPLIANT` | Required PPE observed for this person under current conditions |
| `VIOLATION` | Required PPE confidently absent under current conditions |
| `INDETERMINATE` | Item not observed but detection unreliable for this class, occluded, or the person is at insufficient resolution |

Classes without demonstrated cross-domain reliability return
`INDETERMINATE`, never `VIOLATION`.

---

## 5. Phasing

Ordered by detectability, not by safety importance — the most important item
is fortunately also the most detectable.

| Phase | Classes | Compliance role |
|---|---|---|
| **1 — complete** | Hi-vis vest | Gates the compliance decision |
| **1.5 — next** | Hearing protection (re-scoped from `helmet`); gloves | Advisory only until cross-domain accuracy demonstrated |
| **2** | Footwear presence | Advisory proxy indicator |
| **Deferred** | Eye protection | Pending evidence it is resolvable at operational camera distance |
| **Parallel** | METAR conditional policy layer | Independent of detector work; can be built and tested against synthetic conditions |

**Promotion criterion.** A class moves from advisory to compliance-gating
only after demonstrating agreed accuracy on a held-out *aviation* validation
set — not on the construction-domain validation split, where the original
cross-domain failure was invisible.

---

## 6. Open questions

1. What minimum pixel height on a person is required for reliable
   per-class detection? This determines viable camera placement.
2. Can earmuffs be distinguished from headsets and caps reliably enough to
   be useful, given both are common on ramps?
3. Is a stand-entry close-range camera justified for eye protection, or does
   that PPE class fall outside what apron-wide coverage can support?
4. What is the acceptable false-positive rate before operators disengage?
   This should be established with ramp supervisors, not assumed.
