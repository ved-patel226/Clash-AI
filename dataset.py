from clashroyale import OfficialAPI
from dotenv import load_dotenv
import os
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import multiprocessing


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

    def get_battle_summary(
        self, player_tags, save=True, workers=20, flush_every=1_000, resume=True
    ):

        print("Workers:", workers)

        if isinstance(player_tags, str):
            player_tags = [player_tags]

        summary = {}
        if resume and os.path.exists("battle_summary.json"):
            try:
                with open("battle_summary.json", "r") as f:
                    summary = json.load(f)
            except Exception:
                pass

        pending = [t for t in player_tags if t not in summary]

        card_to_idx = self.card_to_idx  # local ref
        human = self.human_readable

        def process(tag):
            battles = self.api.get_player_battles(tag)
            results = []
            for battle in battles:
                if len(battle.team) != 1 or len(battle.opponent) != 1:
                    # tqdm.write(
                    #     f"Skipping battle with missing team or opponent for player {tag}"
                    # )
                    continue

                team_player = battle.team[0]
                opponent_player = battle.opponent[0]

                deck_team = [
                    (c.name if human else card_to_idx[c.name])
                    for c in team_player.cards
                ]
                deck_opp = [
                    (c.name if human else card_to_idx[c.name])
                    for c in opponent_player.cards
                ]
                results.append(
                    {
                        "team_player": {
                            "tag": team_player.tag,
                            "deck": deck_team,
                            "crowns": team_player.crowns,
                        },
                        "opponent_player": {
                            "tag": opponent_player.tag,
                            "deck": deck_opp,
                            "crowns": opponent_player.crowns,
                        },
                        "result": (
                            "win"
                            if team_player.crowns > opponent_player.crowns
                            else (
                                "loss"
                                if team_player.crowns < opponent_player.crowns
                                else "draw"
                            )
                        ),
                    }
                )
            return tag, results

        if not pending:
            return summary

        last_flush = time.time()
        completed_since_flush = 0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process, tag): tag for tag in pending}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Players"):
                tag, results = (), []
                try:
                    tag, results = fut.result()
                except Exception as e:
                    tqdm.write(f"Error processing {tag}: {e}")
                    continue

                summary[tag] = results
                completed_since_flush += 1
                if save and (
                    completed_since_flush >= flush_every
                    or time.time() - last_flush > 30
                ):
                    with open("battle_summary.json", "w") as f:
                        json.dump(summary, f)
                    last_flush = time.time()
                    completed_since_flush = 0

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
