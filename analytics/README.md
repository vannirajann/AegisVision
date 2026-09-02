# AegisVision - Analytics

This folder contains the analytics modules developed by Person 4.

## Responsibilities

1. Face Detection
2. Night-Time / Low-Light Movement Detection
3. Loitering Detection
4. Basic Suspicious Activity Detection

## Face Detection Flow

Video Frame
    ↓
Face Detection
    ↓
Face Bounding Box
    ↓
Face Event Data

## Night Movement Flow

Video Frame
    ↓
Low-Light / Night Detection
    ↓
Movement Detection
    ↓
Night Movement Event

## Loitering Flow

Tracked Person
    ↓
Person Enters Zone
    ↓
Timer Starts
    ↓
Person Remains Too Long
    ↓
Loitering Alert

## Suspicious Activity Rules

The initial prototype uses simple and explainable rules:

- Person stays too long in a restricted zone.
- Person repeatedly enters a restricted zone.
- Movement occurs during configured night/restricted hours.

## Folder Structure

analytics/
│
├── face/
├── movement/
├── loitering/
├── suspicious/
├── events/
├── tests/
└── README.md

## Development Order

1. Face Detection
2. Movement Detection
3. Low-Light / Night Detection
4. Night Movement Events
5. Loitering Timer
6. Repeated Entry Detection
7. Suspicious Activity Rules
8. Standardized Event Outputs
9. Backend Integration Preparation

## Important

This is an academic prototype.

Face-related testing must use only authorized or consented images and datasets.

The initial face module performs face detection only.

Facial recognition and identity matching are not part of the initial implementation.