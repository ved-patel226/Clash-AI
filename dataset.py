from clashroyale import OfficialAPI
from dotenv import load_dotenv
import os
from tqdm import tqdm
import json


def build_card_name_to_index(card_data):
    """
    Given a card_data dict (name -> info), returns a dict mapping card name to model index.
    The index is the order in which the card appears in card_data.keys().
    """
    return {name: idx for idx, name in enumerate(card_data.keys())}


def get_all_cards(api: OfficialAPI):
    cards = api.get_all_cards()
    card_data = {}
    for card in cards:
        card_info = {
            "name": card["name"],
            "image_url": card["icon_urls"]["medium"],
            "has_evolution": "evolution_medium" in card.get("icon_urls", {}),
            "evolution_image_url": card["icon_urls"].get("evolution_medium", None),
        }
        card_data[card["name"]] = card_info
    return card_data


def get_all_in_clan(api: OfficialAPI, clan_tag: str):
    players = api.get_clan_members(clan_tag)
    return [player.tag for player in players]


class ClashDataset:
    def __init__(self, api: OfficialAPI, human_readable=False):
        self.api = api
        self.human_readable = human_readable
        self.card_to_idx = build_card_name_to_index(get_all_cards(self.api))

    def get_battle_summary(self, player_tags, save=True):
        if isinstance(player_tags, str):
            player_tags = [player_tags]

        summary = {}
        for idx, player_tag in enumerate(tqdm(player_tags, desc="Processing players")):
            try:
                battles = self.api.get_player_battles(player_tag)
                for battle in battles:
                    team_player = battle.team[0]
                    opponent_player = battle.opponent[0]
                    team_crowns = sum(member.crowns for member in battle.team)
                    opponent_crowns = sum(member.crowns for member in battle.opponent)

                    if opponent_crowns > team_crowns:
                        continue

                    result = {
                        "team_player": {
                            "tag": team_player.tag,
                            "deck": [
                                (
                                    card.name
                                    if self.human_readable
                                    else self.card_to_idx[card.name]
                                )
                                for card in team_player.cards
                            ],
                            "crowns": team_player.crowns,
                        },
                        "opponent_player": {
                            "tag": opponent_player.tag,
                            "deck": [
                                (
                                    card.name
                                    if self.human_readable
                                    else self.card_to_idx[card.name]
                                )
                                for card in team_player.cards
                            ],
                            "crowns": opponent_player.crowns,
                        },
                        "result": ("win" if team_crowns > opponent_crowns else "loss"),
                    }
                    if player_tag not in summary:
                        summary[player_tag] = []
                    summary[player_tag].append(result)

                if idx % 100 == 0:
                    with open("battle_summary.json", "w") as f:
                        json.dump(summary, f)
            except Exception as e:
                print(f"Error processing {player_tag}: {e}")

        if save:
            with open("battle_summary.json", "w") as f:
                json.dump(summary, f)
        return summary


if __name__ == "__main__":
    load_dotenv()

    api = OfficialAPI(
        os.getenv("CR_API_KEY"),
    )

    dataset = ClashDataset(api)
    # Read tags from file
    with open("player_tags.txt", "r") as f:
        tags = [line.strip() for line in f.readlines()]

    summary = dataset.get_battle_summary(tags)
