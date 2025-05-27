from carcassone.carcassonne.objects.actions.action import Action
from carcassone.carcassonne.objects.coordinate import Coordinate
from carcassone.carcassonne.objects.tile import Tile


class TileAction(Action):
    def __init__(self, tile: Tile, coordinate: Coordinate, tile_rotations: int):
        self.tile = tile
        self.coordinate = coordinate
        self.tile_rotations = tile_rotations
