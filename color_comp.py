import colorsys

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")  
    if len(hex_color) != 6:
        raise ValueError("Incorrect color format")
    return tuple(int(hex_color[i:i+2], 16) for i in (0,2,4))

def rgbv_to_hex(rgb_color):
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
        palette.append(rgbv_to_hex(comp_rgb))
        
    
    return palette