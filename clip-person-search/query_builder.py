def compose_query(upper_color="", lower_color="", has_backpack=False, has_hat=False, extra=""):
    """
    Assembles a natural language description for CLIP from structured fields.
    Returns at minimum "person".
    """
    parts = []
    if upper_color:
        parts.append(f"{upper_color} shirt")
    if lower_color:
        parts.append(f"{lower_color} pants")
    if has_backpack:
        parts.append("backpack")
    if has_hat:
        parts.append("hat")
    if extra:
        parts.append(extra)

    if not parts:
        return "person"
    return "person with " + " and ".join(parts)


def prompt_query():
    """Asks the user structured questions and returns the composed query string."""
    print("\n--- Describe the suspect ---")
    upper_color  = input("Upper clothing color (e.g. red, blue, black) [Enter to skip]: ").strip()
    lower_color  = input("Lower clothing color (e.g. black, blue) [Enter to skip]: ").strip()
    has_backpack = input("Has backpack? (y/N): ").strip().lower() == "y"
    has_hat      = input("Has hat? (y/N): ").strip().lower() == "y"
    extra        = input("Any other description (e.g. beard, tall) [Enter to skip]: ").strip()

    query = compose_query(upper_color, lower_color, has_backpack, has_hat, extra)
    print(f'\nSearching for: "{query}"\n')
    return query
