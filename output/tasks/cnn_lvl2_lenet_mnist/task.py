    try:
        result = main()
        # Verify quality thresholds
        sys.exit(0)
    except AssertionError as e:
        print(f"Quality check failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
