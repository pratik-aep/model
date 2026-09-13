# Bug Condition Analysis

This document applies the bug condition methodology to the critical bugs identified in the Iceberg Drift Prediction Model.

## Bug Condition Methodology

For each bug, we define:
- **C(X)**: Bug Condition - identifies inputs that trigger the bug
- **P(result)**: Property - desired behavior for buggy inputs
- **¬C(X)**: Non-buggy inputs that should be preserved
- **F**: Original (unfixed) function
- **F'**: Fixed function

---

## BUG-001: Zero Environmental Data

### Bug Condition Function

```pascal
FUNCTION isBugCondition_ZeroEnvData(X)
  INPUT: X of type TrainingDataBatch
  OUTPUT: boolean
  
  // Returns true when environmental data is all zeros
  RETURN (all(X.current_uo = 0.0) AND 
          all(X.current_vo = 0.0) AND 
          all(X.wind_u10 = 0.0) AND 
          all(X.wind_v10 = 0.0))
END FUNCTION
```

### Fix Checking Property

```pascal
// Property: Environmental data must have realistic non-zero values
FOR ALL X WHERE isBugCondition_ZeroEnvData(X) DO
  data' ← generate_or_load_environmental_data'(X)
  
  ASSERT mean(abs(data'.current_uo)) >= 0.1 AND mean(abs(data'.current_uo)) <= 0.5
  ASSERT mean(abs(data'.current_vo)) >= 0.1 AND mean(abs(data'.current_vo)) <= 0.5
  ASSERT mean(abs(data'.wind_u10)) >= 5.0 AND mean(abs(data'.wind_u10)) <= 15.0
  ASSERT mean(abs(data'.wind_v10)) >= 5.0 AND mean(abs(data'.wind_v10)) <= 15.0
  ASSERT std(data'.current_uo) > 0.01  // Non-zero variance
  ASSERT std(data'.wind_u10) > 1.0     // Non-zero variance
END FOR
```

### Preservation Property

```pascal
// Property: Non-zero environmental data should remain unchanged
FOR ALL X WHERE NOT isBugCondition_ZeroEnvData(X) DO
  data ← generate_or_load_environmental_data(X)
  data' ← generate_or_load_environmental_data'(X)
  
  ASSERT data = data'  // No change to already-valid data
END FOR
```

### Counterexample

```python
# Concrete example demonstrating the bug
batch = load_training_batch(idx=0)
print(batch.features[:, feature_idx_current_uo])  # Output: [0.0, 0.0, 0.0, ...]
print(batch.features[:, feature_idx_wind_u10])     # Output: [0.0, 0.0, 0.0, ...]

# Model prediction with zero environmental data
predictions = model(batch)
position_error = compute_error(predictions, ground_truth)
print(position_error)  # Output: 8247.3 km (FAIL - should be < 100 km)
```

---

## BUG-002: Missing Feature Index Mapping

### Bug Condition Function

```pascal
FUNCTION isBugCondition_MissingFeatureMap(X)
  INPUT: X of type PINNModelConfig
  OUTPUT: boolean
  
  // Returns true when feature_idx_map is missing or None
  RETURN (X.feature_idx_map is None OR 
          not hasattr(X, 'feature_idx_map'))
END FUNCTION
```

### Fix Checking Property

```pascal
// Property: PINN models must have explicit feature index mapping
FOR ALL X WHERE isBugCondition_MissingFeatureMap(X) DO
  model' ← initialize_pinn_model'(X)
  
  ASSERT model'.feature_idx_map is not None
  ASSERT 'current_uo' in model'.feature_idx_map
  ASSERT 'current_vo' in model'.feature_idx_map
  ASSERT 'wind_u10' in model'.feature_idx_map
  ASSERT 'wind_v10' in model'.feature_idx_map
  
  // Verify indices point to correct columns
  features ← get_sample_features()
  current_uo_col ← features[:, model'.feature_idx_map['current_uo']]
  ASSERT column_name(current_uo_col) = 'current_uo'  // Correct mapping
END FOR
```

### Preservation Property

```pascal
// Property: Models with existing feature maps should remain unchanged
FOR ALL X WHERE NOT isBugCondition_MissingFeatureMap(X) DO
  model ← initialize_pinn_model(X)
  model' ← initialize_pinn_model'(X)
  
  ASSERT model.feature_idx_map = model'.feature_idx_map
END FOR
```

### Counterexample

```python
# Concrete example demonstrating the bug
config = load_config()
model = PINNModel(config)  # feature_idx_map=None

# Fallback indexing uses wrong columns
features = torch.randn(32, 45)  # 45 engineered features
current_uo = features[:, -4]  # Fallback: assumes last 4 columns
# But after feature engineering, column -4 is actually 'velocity_magnitude', not 'current_uo'

# Physics loss uses wrong features
physics_loss = model.physics_loss(features)  # FAIL - wrong columns
```

---

## BUG-003: Scheduler Step Crash

### Bug Condition Function

```pascal
FUNCTION isBugCondition_SchedulerCrash(X)
  INPUT: X of type ValidationMetrics
  OUTPUT: boolean
  
  // Returns true when early_stopping_metric is missing
  RETURN (config.early_stopping_metric not in X.keys())
END FUNCTION
```

### Fix Checking Property

```pascal
// Property: Scheduler step must handle missing metrics gracefully
FOR ALL X WHERE isBugCondition_SchedulerCrash(X) DO
  result' ← scheduler_step'(X, config)
  
  ASSERT no_exception_raised(result')
  ASSERT result'.metric_used = X.get(config.early_stopping_metric, X['val_loss'])
END FOR
```

### Preservation Property

```pascal
// Property: Scheduler with valid metrics should behave identically
FOR ALL X WHERE NOT isBugCondition_SchedulerCrash(X) DO
  result ← scheduler_step(X, config)
  result' ← scheduler_step'(X, config)
  
  ASSERT result.metric_used = result'.metric_used
  ASSERT result.learning_rate = result'.learning_rate
END FOR
```

### Counterexample

```python
# Concrete example demonstrating the bug
config.early_stopping_metric = 'val_rmse'
val_metrics = {'val_loss': 0.5, 'val_mae': 12.3}  # Missing 'val_rmse'

# Original code crashes
try:
    metric = val_metrics[config.early_stopping_metric]  # KeyError!
except KeyError:
    print("CRASH - KeyError: 'val_rmse'")
```

---

## BUG-006: Constant Target Values

### Bug Condition Function

```pascal
FUNCTION isBugCondition_ConstantTargets(X)
  INPUT: X of type TrajectorySegment
  OUTPUT: boolean
  
  // Returns true when velocity targets have no variation
  RETURN (std(X.v_velocity_target) < 0.001 OR 
          all_equal(X.v_velocity_target))
END FUNCTION
```

### Fix Checking Property

```pascal
// Property: Training segments must have varying targets
FOR ALL X WHERE isBugCondition_ConstantTargets(X) DO
  segments' ← filter_training_segments'(X)
  
  // Constant segments should be filtered out
  ASSERT X not in segments'
  
  // OR handled separately
  ASSERT (X in segments_stationary) AND (X not in segments_training)
END FOR
```

### Preservation Property

```pascal
// Property: Segments with varying targets should remain in training
FOR ALL X WHERE NOT isBugCondition_ConstantTargets(X) DO
  segments ← filter_training_segments(X)
  segments' ← filter_training_segments'(X)
  
  ASSERT (X in segments) = (X in segments')
END FOR
```

### Counterexample

```python
# Concrete example demonstrating the bug
segment = load_trajectory_segment(iceberg_id=42, start_idx=100)
print(segment['v_velocity'])  
# Output: [49.9952, 49.9952, 49.9952, ...] (20 identical values)

# Model training produces zero gradient
loss = criterion(model(segment.features), segment.targets)
loss.backward()
print(model.lstm.weight_hh.grad)  # Output: near-zero gradients - no learning!
```

---

## BUG-007: Missing Data Quality Checks

### Bug Condition Function

```pascal
FUNCTION isBugCondition_InvalidData(X)
  INPUT: X of type EnvironmentalDataset
  OUTPUT: boolean
  
  // Returns true when data fails quality checks
  RETURN (has_nan(X) OR 
          has_zero_variance(X) OR 
          values_out_of_range(X))
END FUNCTION
```

### Fix Checking Property

```pascal
// Property: Data validation must detect and reject invalid data
FOR ALL X WHERE isBugCondition_InvalidData(X) DO
  result' ← validate_data_quality'(X)
  
  ASSERT result'.is_valid = False
  ASSERT len(result'.errors) > 0
  ASSERT result'.errors contains descriptive_message
  
  // Training should not proceed
  ASSERT_RAISES(DataQualityError, train_model'(X))
END FOR
```

### Preservation Property

```pascal
// Property: Valid data should pass validation unchanged
FOR ALL X WHERE NOT isBugCondition_InvalidData(X) DO
  result ← validate_data_quality'(X)
  
  ASSERT result.is_valid = True
  ASSERT len(result.errors) = 0
  ASSERT train_model'(X) proceeds_normally
END FOR
```

### Counterexample

```python
# Concrete example demonstrating the bug
data = load_environmental_data()
print(np.isnan(data['current_uo']).sum())  # Output: 523 NaN values
print(data['wind_u10'].std())               # Output: 0.0 (zero variance)

# No validation - garbage silently enters training
train_model(data)  # Produces garbage predictions without warning
```

---

## Success Criteria

The bugfixes are validated when:

1. **Fix Checking**: All buggy inputs (C(X) = True) now produce correct behavior according to property P
2. **Preservation Checking**: All non-buggy inputs (¬C(X)) continue to work identically (F(X) = F'(X))
3. **System Metrics**: 
   - Position RMSE < 100 km (currently 8000+ km)
   - Skill score vs persistence > 0 (currently -443)
   - Training converges within 100 epochs
   - No crashes or dimension errors

4. **Data Quality Metrics**:
   - All environmental features have non-zero mean and variance
   - Feature ranges within expected physical bounds
   - No NaN or infinite values in training data

5. **Property-Based Tests Pass**:
   - Test with randomly generated inputs satisfying C(X)
   - Verify P(result) holds for 1000+ generated cases
   - Verify F(X) = F'(X) for 1000+ non-buggy cases
