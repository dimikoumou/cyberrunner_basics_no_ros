try:
    from pyocamcalib.modelling.calibration import CalibrationEngine
    print("Successfully imported CalibrationEngine")
except ImportError as e:
    print(f"Import failed: {e}")

