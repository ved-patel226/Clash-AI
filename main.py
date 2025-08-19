from clashroyale import OfficialAPI, RoyaleAPI
from dotenv import load_dotenv
import os
import requests
from PIL import Image
import io

load_dotenv()

os.makedirs("decks", exist_ok=True)


class DeckVisualizer:
    def __init__(self, api: OfficialAPI):
        self.api = api

    def fetch_and_save_decks(self, top_n_clans=1, members_per_clan=1):
        clans_data = self.api.get_top_clanwar_clans()
        for n, clan in enumerate(clans_data):
            print(f"{n+1}. {clan.name} - {clan.clan_score} - {clan.tag}")
            clan_details = self.api.get_clan(clan.tag)
            print("   Members:")
            self._process_clan_members(clan_details.member_list, members_per_clan)
            print()
            if n + 1 >= top_n_clans:
                break

    def _process_clan_members(self, member_list, members_per_clan):
        for m, member in enumerate(member_list):
            if m >= members_per_clan:
                break
            player_data = self.api.get_player(member.tag)
            deck = player_data["currentDeck"]
            images = self._download_deck_images(deck)
            combined_img = self._combine_images(images)
            filename = self._get_deck_filename(member.name)
            combined_img.save(filename)
            print(f"      Saved deck image: {filename}")

    def _download_deck_images(self, deck):
        images = []
        for card in deck:
            if (
                card.get("evolution_level", 0) > 0
                and "evolution_medium" in card["icon_urls"]
            ):
                img_url = card["icon_urls"]["evolution_medium"]
            else:
                img_url = card["icon_urls"]["medium"]
            response = requests.get(img_url)
            img = Image.open(io.BytesIO(response.content))
            images.append(img)
        return images

    def _combine_images(self, images):
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        combined_img = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            combined_img.paste(img, (x_offset, 0))
            x_offset += img.width
        return combined_img

    def _get_deck_filename(self, member_name):
        return f"./decks/{member_name.replace(' ', '_').replace('/', '_')}_deck.png"


if __name__ == "__main__":
    api = RoyaleAPI(os.getenv("CR_API_KEY"))

    print(api.get_top_players())
