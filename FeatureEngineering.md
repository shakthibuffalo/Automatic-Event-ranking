# Feature Generation Notebook

## Overview

This notebook demonstrates the feature engineering steps performed on sensor data collected around vehicle events. Each feature captures specific driver behaviors or sensor metrics relevant to classifying event severity and driver performance.

---

## Data Preparation

### Import Libraries

```python
import pandas as pd
import numpy as np
```

### Data Structure
- **Main Table (`t`)**: Metrics at event occurrence.
- **PrePostEvent (`p`)**: Metrics collected every 0.25 sec within ±10 seconds around events.

---

## Important Feature Engineering Steps

### 1. Ratio of Speed Change

Measures driver's speed adjustment before vs. after an event.

```python
def RatioOfSpeedChange(df):
    # Speeds before and after event
    df_before_zero = df[df['SecTimeSegment'] < 0].groupby(['DriverID', 'TripID', 'TripEventID'])['ppeSpeed'].agg(['first', 'last']).reset_index()
    df_after_zero = df[df['SecTimeSegment'] >= 0].groupby(['DriverID', 'TripID', 'TripEventID'])['ppeSpeed'].agg(['first', 'last']).reset_index()

    df_before_zero['DeltaSpeedBeforeZero'] = df_before_zero['first'] - df_before_zero['last']
    df_after_zero['DeltaSpeedAfterZero'] = df_after_zero['first'] - df_after_zero['last']

    combined = pd.merge(df_before_zero, df_after_zero, on=['DriverID', 'TripID', 'TripEventID'], suffixes=('_before', '_after'))

    combined['DeltaSpeedRatio'] = combined['DeltaSpeedBeforeZero'] / (0.001 + combined['DeltaSpeedAfterZero'])
    return combined
```

**Behavior captured**: Reflects how abruptly the driver changes speed in response to events, indicating severity or driver responsiveness.

---

### 2. Braking Force Duration

Identifies significant braking intervals surrounding an event.

```python
def process_dataframe(df):
    positive = df[df['AbsValBrakingForce'] > 0.1].groupby('TripEventID')['SecTimeSegment'].agg(['first', 'last']).reset_index()
    negative = df[df['AbsValBrakingForce'] <= 0.1][['TripEventID']].drop_duplicates()
    negative['first'], negative['last'] = 10, -10

    union = pd.concat([positive, negative]).groupby('TripEventID').agg(Min_SecTimeSegment=('first','min'), Max_SecTimeSegment=('last','max')).reset_index()
    return union
```

**Behavior captured**: Indicates time frames of intense braking activity, highlighting sudden or prolonged braking behavior.

---

### 3. Braking Irregularity

Detects inconsistency in braking behavior.

```python
def BrakingIrregularity(df):
    df['BFCenterDeviation'] = np.where(df['DriverID'] == df['DriverID'].shift(1),
                                       abs(df.ppeBrakingForce.shift(1) - 0.5*(df.ppeBrakingForce + df.ppeBrakingForce.shift(2))), np.nan)

    grouped = df.groupby('TripEventID')['BFCenterDeviation'].agg(['sum','max']).reset_index()
    return grouped
```

**Behavior captured**: High values represent erratic braking, indicating uncertainty or panic.

---

### 4. Turning Force Irregularity

Captures sudden or inconsistent steering.

```python
def TurningIrregularity(df):
    df['TFCenterDeviation'] = np.where(df['DriverID'] == df['DriverID'].shift(1),
                                       abs(df.ppeTurningForce.shift(1) - 0.5*(df.ppeTurningForce + df.ppeTurningForce.shift(2))), np.nan)

    grouped = df.groupby('TripEventID')['TFCenterDeviation'].agg(['sum','max']).reset_index()
    return grouped
```

**Behavior captured**: High irregularity indicates abrupt or aggressive turning, signaling risky behavior.

---

### 5. Post-Event Speed Fluctuation Count

Counts frequency of significant speed changes after the event.

```python
def sppedcountafter0(df):
    df['SpeedChange'] = abs(df.ppeSpeed - df.ppeSpeed.shift(1))
    post_event = df[(df['SecTimeSegment'] >= 0) & (df['SpeedChange'] > 0)]
    counts = post_event.groupby('TripEventID')['SpeedChange'].agg('count').reset_index().rename(columns={'SpeedChange':'SpeedChangeCount'})
    return counts
```

**Behavior captured**: Frequent post-event speed adjustments signal instability.

---

## Aggregation & Final Features

Combine all engineered features into a final feature set per event:

```python
final_features = df.groupby(['CompanyID', 'TripEventID']).agg({
    'AbsVal_TF_Moving_Delta':['mean','sum','std'],
    'AbsDiffBrakingForce':['mean','sum','std','max'],
    'DeltaSpeedRatio':'max',
    'AfterZeroBrakeOnCount':'first',
    'Sum_CenterDifferenceDeviation':'max',
    'Max_CenterDifferenceDeviation':'max',
    'Sum_TFCenterDeviation':'max',
    'Max_AbsValTurningForce':'first',
    'Max_AbsValBrakingForce':'first',
    'Max_BFCenterDeviation':'max',
    'isSevere':'first'
}).reset_index()

final_features.columns = ['_'.join(col).strip() for col in final_features.columns.values]
```

These aggregated features provide a comprehensive representation of driver behaviors, ideal for predictive modeling of event severity and driver performance.

---

## Behavioral Insights Captured:
- **Speed adjustments**: Reaction time and driver response severity.
- **Braking intensity and irregularity**: Captures suddenness, panic braking, and inconsistency.
- **Turning behavior**: Indicates aggression or loss of control.
- **Speed fluctuation frequency post-event**: Indicates continued instability or panic.

These engineered features greatly enhance predictive capability, ensuring robust model performance.

