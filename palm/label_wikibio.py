# %%
# save model outputs into running pandas df, storing all parameters of the generation
import vertexai
from vertexai.language_models import TextGenerationModel
import pandas as pd
from unidecode import unidecode
from functional import seq
from pathlib import Path
import hashlib
import time
# %%
PROMPT = lambda text : f"""\'mass, followedBy, champions, formerTeam, spouse, award, order, militaryBranch, ISSN_number, academicStaffSize, escapeVelocity, fullname, powerType, foundationPlace, ISBN_number, floorCount, abbreviation, populationDensity, precededBy, spokenIn, currency, activeYearsStartYear, countryOrigin, origin, season, completionDate, was selected by NASA, birthYear, musicFusionGenre, length, course, derivative, assembly, instrument, mediaType, alternativeName, areaTotal, commander, status, architect, publisher, operator, owner, runwayName, author, numberOfMembers, affiliation, periapsis, background, recordLabel, launchSite, relatedMeanOfTransportation, established, discoverer, leader, largestCity, engine, orbitalPeriod, apoapsis, builder, party, state, manufacturer, league, demonym, occupation, was a crew member of, elevationAboveTheSeaLevel_(in_metres), office (workedAt,  workedAs), dishVariation, creator, operatingOrganisation, successor, battles, epoch, stylisticOrigin, cityServed, mainIngredients, birthDate, runwayLength, city, leaderTitle, nationality, almaMater, manager, ground, deathPlace, region, ingredient, associatedBand, capital, genre, language, ethnicGroup, club, isPartOf, leaderName, location, birthPlace, country\'

Based on the example relations above, what relations are inferable from the following sentence? Answer in the format
[\"Noun | relation | noun2\",
\"Noun3 | relation2 | noun4\"]

input: aarhus airport\'s runway length is 2702.0.
output:
<sentence>
Aarhus Airport\'s runway length is 2702.0
</sentence>
<labels>
["Aarhus Airport | runwayLength | 2702.0"]
</labels>
<DONE>

input: torvalds was born in helsinki, finland, the son of journalists anna and nils torvalds, the grandson of statistician leo törnqvist and of poet ole torvalds, and the great-grandson of journalist and soldier toivo karanko.
output:
<sentence>
Torvalds was born in Helsinki, Finland, the son of journalists Anna and Nils Torvalds, the grandson of statistician Leo Törnqvist and of poet Ole Torvalds, and the great-grandson of journalist and soldier Toivo Karanko.
</sentence>
<labels>
["Torvalds | birthPlace | Helsinki, Finland",
"Torvalds | father | Nils Torvalds",
"Torvalds | mother | Anna Torvalds",
"Torvalds | paternalGrandfather | Leo Törnqvist",
"Torvalds | maternalGrandfather | Ole Torvalds",
"Torvalds | maternalGreatGrandfather | Toivo Karanko"]
</labels>
<DONE>

input: {text}
output:"""

vertexai.init(project="deception-emotion", location="us-central1")

def retrieve_generation(text, i):
    print("progress ", i, text)
    uuid = hashlib.sha256(text.encode('utf-8')).hexdigest()
    generation_prior = Path("generations/" + uuid)
    if generation_prior.exists():
        return generation_prior.read_text()
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "stop_sequences": [
            "<DONE>"
        ],
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40
    }
    model = TextGenerationModel.from_pretrained("text-bison")
    time.sleep(10)
    try:
        response = model.predict(PROMPT(text), **parameters)
        print(f"Response from Model: {response.text}")
        generation_prior.write_text( f"<id>{i}</id>" + response.text)
        return response.text
    except Exception as e:
        print(e)

# %%

if __name__ == "__main__":
    df = pd.read_pickle("~/repos/nlgs-research/pipeline/normalized_data/wikibio.pkl").head(5000)
    res = seq(df.target_text).zip(df.index).starmap(retrieve_generation)
    print(res)
# %%
