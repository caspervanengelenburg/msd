from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors


# SUBTYPE MAPPINGS
ZONING_MAPPING = {
    'ROOM': 'Zone1',
    'BEDROOM': 'Zone1',
    'BALCONY': 'Zone4',
    'KITCHEN': 'Zone2',
    'DINING': 'Zone2',
    'KITCHEN_DINING': 'Zone2',
    'LIVING_ROOM': 'Zone2',
    'LIVING_DINING': 'Zone2',
    'BATHROOM': 'Zone3',
    'CORRIDOR': 'Zone2',
    'CORRIDORS_AND_HALLS': 'Zone2',
    'STAIRS': 'Zone3',
    'STAIRCASE': 'Zone3',
    'ELEVATOR': 'Zone3',
    'RAILING': 'Structure',
    'VOID': 'Zone3',
    'SHAFT': 'Structure',
    'WALL': 'Structure',
    'COLUMN': 'Structure',
    'STOREROOM': 'Zone3',
    'ENTRANCE_DOOR': 'Entrance Door',
    'DOOR': 'Door',
    'WINDOW': 'Window'
}

ROOM_MAPPING = {
    'ROOM': 'Bedroom',  # ZONE 1
    'BEDROOM': 'Bedroom',  # ZONE 1
    'BALCONY': 'Balcony',  # ZONE 4
    'KITCHEN': 'Kitchen',  # ZONE 2
    'DINING': 'Dining', # ZONE 2
    'KITCHEN_DINING': 'Kitchen', # ZONE 2
    'LIVING_ROOM': 'Livingroom', # ZONE 2
    'LIVING_DINING': 'Livingroom', # ZONE 2
    'BATHROOM': 'Bathroom', # ZONE 3
    'CORRIDOR': 'Corridor', # ZONE 2
    'CORRIDORS_AND_HALLS': 'Corridor', # ZONE 2
    'STAIRS': 'Stairs', # ZONE 3
    'STAIRCASE': 'Stairs', # ZONE 3
    'ELEVATOR': 'Stairs', # ZONE 3
    'RAILING': 'Structure', # Structure
    'VOID': 'Stairs', # ZONE 3
    'SHAFT': 'Structure', # Structure
    'WALL': 'Structure', # Structure
    'COLUMN': 'Structure', # Structure
    'STOREROOM': 'Storeroom', # ZONE 3
    'ENTRANCE_DOOR': 'Entrance Door', # Entrance Door
    'DOOR': 'Door', # Door
    'WINDOW': 'Window' # Window
}


ZONING_NAMES = ['Zone1', 'Zone2', 'Zone3', 'Zone4', 'Structure', 'Door', 'Entrance Door', 'Window']
ROOM_NAMES = ['Bedroom', 'Livingroom', 'Kitchen', 'Dining', 'Corridor', 'Stairs', 'Storeroom', 'Bathroom', 
              'Balcony', 'Structure', 'Door', 'Entrance Door', 'Window']
ZONING_ROOMS = {'Zone1': ['Bedroom'],
                'Zone2': ['Livingroom', 'Kitchen', 'Dining', 'Corridor'],
                'Zone3': ['Stairs', 'Storeroom', 'Bathroom'],
                'Zone4': ['Balcony'],
                'Structure': ['Structure'],
                'Door': ['Door'],
                'Entrance Door': ['Entrance Door'],
                'Window': ['Window']}


# COLORING
COLORS_ZONING = ['#1f77b4',
                  '#ff7f0e',
                  '#72246c',
                  '#2ca02c',
                  '#000000',
                  '#1f77b4',
                  '#98df8a',
                  '#d62728']

COLOR_MAP_ZONING = mcolors.ListedColormap(COLORS_ZONING)
CMAP_ZONING = get_cmap(COLOR_MAP_ZONING)


COLORS_ROOMTYPE = ['#1f77b4',
                      '#e6550d',
                      '#fd8d3c',
                      '#87ceeb',
                      '#ffff00',
                      '#72246c',
                      '#5254a3',
                      '#6b6ecf',
                      '#2ca02c',
                      '#000000',
                      '#1f77b4',
                      '#98df8a',
                      '#d62728']

COLOR_MAP_ROOMTYPE = mcolors.ListedColormap(COLORS_ROOMTYPE)
CMAP_ROOMTYPE = get_cmap(COLOR_MAP_ROOMTYPE)