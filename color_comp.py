import colorsys

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")  
    if len(hex_color) != 6:
        raise ValueError("Incorrect color format")
    return tuple(int(hex_color[i:i+2], 16) for i in (0,2,4))

def rgb_to_hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])


def get_comp_color(hex_color, candidates_list = None):
    """
        Given a hex colour, output its complement colours in a ranking
    """
    
    rgb = hex_to_rgb(hex_color)
    rgb_norm = tuple(channel / 255.0 for channel in rgb)
    
    # Convert rgb to HLS (Hue, lightness and saturation)
    h, l, s = colorsys.rgb_to_hls(*rgb_norm)
    compute_hue = (h + 0.5) % 1.0
    
    
    # I coppied this part from GPT but this is basically a default setting 
    if candidates_list is None:
        candidates_list = [
            (0.50, 1.00),  # Rank 1: Vibrant, mid brightness
            (0.60, 0.90),  # Rank 2: Slightly brighter and a bit softer
            (0.40, 1.00),  # Rank 3: Slightly darker and vivid
            (0.50, 0.80),  # Rank 4: Mid brightness but less saturated
            (0.55, 0.85)   # Rank 5: In-between option
        ]
    
    palette = []
    
    for cand_l, cand_s in candidates_list:
        comp_rgb_norm = colorsys.hls_to_rgb(compute_hue, cand_l, cand_s)
        comp_rgb = tuple(int(round(channel *255)) for channel in comp_rgb_norm)
        palette.append(rgb_to_hex(comp_rgb))
        
    
    return palette

def get_analogous_palette(hex_color, candidates_list = None, offset = 0.0833):
    """
        Generate to analogous palettes ( they usually share a common hue and blend well together)
    """
    rgb = hex_to_rgb(hex_color)
    rgb = hex_to_rgb(hex_color)
    rgb_norm = tuple(ch / 255.0 for ch in rgb)
    h,_,_ = colorsys.rgb_to_hls(*rgb_norm)
    
    if candidates_list is None:
        candidates_list = [
            (0.50, 1.00),
            (0.60, 0.90),
            (0.40, 1.00),
            (0.50, 0.80),
            (0.55, 0.85)
        ]

    left_palette = []
    right_palette = []
    for cand_l, cand_s in candidates_list:
        left_hue = (h - offset) % 1.0
        right_hue = (h + offset) % 1.0
        left_rgb_norm = colorsys.hls_to_rgb(left_hue, cand_l, cand_s)
        right_rgb_norm = colorsys.hls_to_rgb(right_hue, cand_l, cand_s)
        left_rgb = tuple(int(round(ch * 255)) for ch in left_rgb_norm)
        right_rgb = tuple(int(round(ch * 255)) for ch in right_rgb_norm)
        left_palette.append(rgb_to_hex(left_rgb))
        right_palette.append(rgb_to_hex(right_rgb))
    
    return left_palette, right_palette

def get_triadic_palette(hex_color, candidates_list = None):
    
    """
        Generates triadic palette (Triadic colors are three colors that are evenly spaced around the color wheel, forming a triangle)
    """
    
    
    rgb = hex_to_rgb(hex_color)
    rgb = hex_to_rgb(hex_color)
    rgb_norm = tuple(ch / 255.0 for ch in rgb)
    h,_,_ = colorsys.rgb_to_hls(*rgb_norm)
    
    hue_1 = ( h + 1/3 ) % 1.0
    hue_2 = ( h + 2/3 ) % 1.0
    
    if candidates_list is None:
        candidates_list = [
            (0.50, 1.00),
            (0.60, 0.90),
            (0.40, 1.00),
            (0.50, 0.80),
            (0.55, 0.85)
        ]
        
    triadic1 = []
    triadic2 = []
    
    for cand_l, cand_s in candidates_list:
        rgb1_norm = colorsys.hls_to_rgb(hue_1, cand_l, cand_s)
        rgb2_norm = colorsys.hls_to_rgb(hue_2, cand_l, cand_s)
        rgb1 = tuple(int(round(ch * 255)) for ch in rgb1_norm)
        rgb2 = tuple(int(round(ch * 255)) for ch in rgb2_norm)
        triadic1.append(rgb_to_hex(rgb1))
        triadic2.append(rgb_to_hex(rgb2))
    
    return triadic1, triadic2

def get_split_comp_palette(hex_color, candidates_list = None, offset = 0.0833):
    
    """
        Generate split complementary color (A split-complementary scheme uses one main color and the two colors adjacent to its complementary color)
    """
    rgb = hex_to_rgb(hex_color)
    rgb_norm = tuple(ch / 255.0 for ch in rgb)
    h, _, _ = colorsys.rgb_to_hls(*rgb_norm)
    
    hue_left = (h + 0.5 - offset) % 1.0
    hue_right = (h + 0.5 + offset) % 1.0
    
    if candidates_list is None:
        candidates_list = [
            (0.50, 1.00),
            (0.60, 0.90),
            (0.40, 1.00),
            (0.50, 0.80),
            (0.55, 0.85)
        ]
    
    split_palette_left = []
    split_palette_right = []
    for cand_l, cand_s in candidates_list:
        rgb_left_norm = colorsys.hls_to_rgb(hue_left, cand_l, cand_s)
        rgb_right_norm = colorsys.hls_to_rgb(hue_right, cand_l, cand_s)
        rgb_left = tuple(int(round(ch * 255)) for ch in rgb_left_norm)
        rgb_right = tuple(int(round(ch * 255)) for ch in rgb_right_norm)
        split_palette_left.append(rgb_to_hex(rgb_left))
        split_palette_right.append(rgb_to_hex(rgb_right))
    
    return split_palette_left, split_palette_right

def get_all(hex_color):
    """
        Returns a dictionary of recommended colors
    """
    
    palettes = {
        "complementary": get_comp_color(hex_color)
    }
    
    analogous_left, analogous_right = get_analogous_palette(hex_color)
    palettes["analogous_left"] = analogous_left
    palettes["analogous_right"] = analogous_right
    
    triadic1, triadic2 = get_triadic_palette(hex_color)
    palettes["triadic1"] = triadic1
    palettes["triadic2"] = triadic2

    split_left, split_right = get_split_comp_palette(hex_color)
    palettes["split_complementary_left"] = split_left
    palettes["split_complementary_right"] = split_right

    return palettes