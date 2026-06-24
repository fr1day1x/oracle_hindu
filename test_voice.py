import outetts

print("Testing direct import structure...")
try:
    # Let's inspect what attributes are directly exposed on the module root
    print("Available top-level tools:", [attr for attr in dir(outetts) if not attr.startswith("_")])
    
    # Try importing their structural types directly from the top level module mapping
    from outetts import Backend, GGUFQuantization, Models
    print("Success! Enums found at top-level.")
except ImportError as e:
    print(f"Top-level import skipped: {e}")
    # Fallback to absolute relative mapping addresses
    from outetts.interface import Backend
    from outetts.models.config import GGUFQuantization
    print("Fallback successful!")

print("If this script runs without dropping, we can use these mappings safely!")