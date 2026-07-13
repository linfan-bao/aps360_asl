# ASL Fingerspelling — Self-Collected Test Set Protocol

## Purpose
A never-before-seen test set of real ASL hand gestures, collected by us and kept
fully separate from the Kaggle training/validation data. Used to test how well the
model generalizes to real-world conditions.

## Plan at a glance
- People to recruit: 5
- Each person shoots the full alphabet (A–Z) — no need to track who shot which letters
- Photos per letter, per person: 3
- Photos per person: 26 × 3 = 78
- Total: 5 × 78 = 390  (= 15 images per class)
- Each person uses their own lighting and background — that is where the real-world
  variety comes from

## Per-person session (same steps every time you recruit someone)
1. Consent — tell them it's for a course project, learning use only, and only the hand
   and wrist will be in frame. Get a verbal yes.
2. Setup — their own room and lighting; a phone camera is fine.
3. Shoot — go through A to Z. For each letter, form the handshape and take 3 photos,
   slightly changing the angle/distance between shots.
4. Framing — keep only the hand and wrist in frame. No face, no ID.
5. Save — drop the photos into data/selfcollected/<LETTER>/ (one folder per letter).

## Folder / naming
data/selfcollected/
  A/  p1_1.jpg  p1_2.jpg  p1_3.jpg  p2_1.jpg ...
  B/  ...
  ...
  Z/
p1 = person 1, etc. Same folder-per-class layout as the training data, so it loads
the same way later.

## Ethics
- Only hand and wrist are captured — no faces or identifying information.
- Participants are told the purpose (academic, learning only) and give consent.
- Data is used for this course project only.

## Handshape reference
Use a standard ASL alphabet chart so every handshape is correct and matches its label.

## Progress report vs final
- This week (progress report): only this plan + 3–5 sample photos are needed.
- Full 390-photo collection: for the final report.
