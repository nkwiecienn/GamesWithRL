from carcassone.carcassonne.carcassonne_game_state import CarcassonneGameState
from carcassone.carcassonne.objects.actions.tile_action import TileAction
from carcassone.carcassonne.objects.coordinate import Coordinate
from carcassone.carcassonne.objects.actions.meeple_action import MeepleAction
from carcassone.carcassonne.objects.coordinate_with_side import CoordinateWithSide
from carcassone.carcassonne.objects.farm import Farm
from carcassone.carcassonne.objects.farmer_connection import FarmerConnection
from carcassone.carcassonne.objects.farmer_connection_with_coordinate import FarmerConnectionWithCoordinate
from carcassone.carcassonne.objects.meeple_position import MeeplePosition
from carcassone.carcassonne.objects.meeple_type import MeepleType
from carcassone.carcassonne.objects.side import Side
from carcassone.carcassonne.objects.terrain_type import TerrainType
from carcassone.carcassonne.objects.tile import Tile
from carcassone.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from carcassone.carcassonne.utils.city_util import CityUtil
from carcassone.carcassonne.utils.farm_util import FarmUtil
from carcassone.carcassonne.utils.road_util import RoadUtil


class PossibleMoveFinder:

    @classmethod
    def possible_meeple_actions(cls, game_state: CarcassonneGameState) -> [MeepleAction]:
        current_player = game_state.current_player
        last_tile_action: TileAction = game_state.last_tile_action
        last_played_tile: Tile = last_tile_action.tile
        last_played_position: Coordinate = last_tile_action.coordinate

        possible_actions: [MeepleAction] = []

        meeple_positions = cls.__possible_meeple_positions(game_state=game_state)

        if SupplementaryRule.FARMERS in game_state.supplementary_rules:
            farmer_positions = cls.__possible_farmer_position(game_state=game_state)
        else:
            farmer_positions = ()

        if game_state.meeples[current_player] > 0:
            possible_actions.extend(list(
                map(lambda x: MeepleAction(meeple_type=MeepleType.NORMAL, coordinate_with_side=x), meeple_positions)))

            possible_actions.extend(list(
                map(lambda x: MeepleAction(meeple_type=MeepleType.FARMER, coordinate_with_side=x), farmer_positions)))

        if game_state.big_meeples[current_player] > 0:
            possible_actions.extend(
                list(map(lambda x: MeepleAction(meeple_type=MeepleType.BIG, coordinate_with_side=x), meeple_positions)))

            possible_actions.extend(list(
                map(lambda x: MeepleAction(meeple_type=MeepleType.BIG_FARMER, coordinate_with_side=x), farmer_positions)))

        if game_state.abbots[current_player] > 0:
            if last_played_tile.chapel or last_played_tile.flowers:
                possible_actions.append(MeepleAction(meeple_type=MeepleType.ABBOT,
                                                     coordinate_with_side=CoordinateWithSide(
                                                         coordinate=last_played_position, side=Side.CENTER)))

        placed_meeple: MeeplePosition
        for placed_meeple in game_state.placed_meeples[current_player]:
            if placed_meeple.meeple_type == MeepleType.ABBOT:
                possible_actions.append(
                    MeepleAction(meeple_type=MeepleType.ABBOT, coordinate_with_side=placed_meeple.coordinate_with_side,
                                 remove=True))

        return possible_actions

    @staticmethod
    def __possible_meeple_positions(game_state: CarcassonneGameState) -> [CoordinateWithSide]:
        playing_positions: [CoordinateWithSide] = []
        last_tile_action: TileAction = game_state.last_tile_action
        last_played_tile: Tile = last_tile_action.tile
        last_played_position: Coordinate = last_tile_action.coordinate

        if last_played_tile.chapel:
            playing_positions.append(CoordinateWithSide(coordinate=last_played_position, side=Side.CENTER))

        if last_played_tile.flowers \
                and SupplementaryRule.NORMAL_MEEPLES_CAN_USE_FLOWERS in game_state.supplementary_rules:
            playing_positions.append(CoordinateWithSide(coordinate=last_played_position, side=Side.CENTER))

        for side in [Side.TOP, Side.RIGHT, Side.BOTTOM, Side.LEFT]:
            if last_played_tile.get_type(side) == TerrainType.CITY:
                connected_cities = CityUtil.find_city(
                    game_state,
                    CoordinateWithSide(coordinate=last_played_position, side=side)
                )
                if CityUtil.city_contains_meeples(game_state, connected_cities):
                    continue
                else:
                    playing_positions.append(CoordinateWithSide(coordinate=last_played_position, side=side))

            if last_played_tile.get_type(side) == TerrainType.ROAD:
                connected_roads = RoadUtil.find_road(
                    game_state,
                    CoordinateWithSide(coordinate=last_played_position, side=side)
                )
                if RoadUtil.road_contains_meeples(game_state, connected_roads):
                    continue
                else:
                    playing_positions.append(CoordinateWithSide(coordinate=last_played_position, side=side))

        return playing_positions

    @classmethod
    def __possible_farmer_position(cls, game_state: CarcassonneGameState) -> [CoordinateWithSide]:
        playing_positions: [CoordinateWithSide] = []
        last_tile_action: TileAction = game_state.last_tile_action
        last_played_tile: Tile = last_tile_action.tile
        last_played_position: Coordinate = last_tile_action.coordinate

        farmer_connection: FarmerConnection
        for farmer_connection in last_played_tile.farms:
            farm: Farm = FarmUtil.find_farm(
                game_state=game_state,
                farmer_connection_with_coordinate=FarmerConnectionWithCoordinate(farmer_connection, last_played_position)
            )
            if FarmUtil.has_meeples(game_state, farm):
                continue
            else:
                farmer_position: Side = farmer_connection.farmer_positions[0]
                playing_positions.append(CoordinateWithSide(last_played_position, farmer_position))

        return playing_positions
