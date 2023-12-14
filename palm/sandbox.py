# %%
# save model outputs into running pandas df, storing all parameters of the generation
import vertexai
from vertexai.language_models import TextGenerationModel
import pandas as pd
from unidecode import unidecode

df = pd.read_pandas("~/repos/nlgs-research/pipeline/normalized_data/wikibio.pkl")
df
# %%

vertexai.init(project="deception-emotion", location="us-central1")
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
response = model.predict(
    """\'mass, followedBy, champions, formerTeam, spouse, award, order, militaryBranch, ISSN_number, academicStaffSize, escapeVelocity, fullname, powerType, foundationPlace, ISBN_number, floorCount, abbreviation, populationDensity, precededBy, spokenIn, currency, activeYearsStartYear, countryOrigin, origin, season, completionDate, was selected by NASA, birthYear, musicFusionGenre, length, course, derivative, assembly, instrument, mediaType, alternativeName, areaTotal, commander, status, architect, publisher, operator, owner, runwayName, author, numberOfMembers, affiliation, periapsis, background, recordLabel, launchSite, relatedMeanOfTransportation, established, discoverer, leader, largestCity, engine, orbitalPeriod, apoapsis, builder, party, state, manufacturer, league, demonym, occupation, was a crew member of, elevationAboveTheSeaLevel_(in_metres), office (workedAt,  workedAs), dishVariation, creator, operatingOrganisation, successor, battles, epoch, stylisticOrigin, cityServed, mainIngredients, birthDate, runwayLength, city, leaderTitle, nationality, almaMater, manager, ground, deathPlace, region, ingredient, associatedBand/associatedMusicalArtist, capital, genre, language, ethnicGroup, club, isPartOf, leaderName, location, birthPlace, country\'

Based on the example relations above, what relations are inferable from the following sentence? Answer in the format
[\"Noun | relation | noun2\",
\"Noun3 | relation2 | noun4\"]

input: Aarhus Airport\'s runway length is 2702.0.
output: Aarhus Airport|runwayLength|2702.0 <DONE>

input: Torvalds was born in Helsinki, Finland, the son of journalists Anna and Nils Torvalds, the grandson of statistician Leo Törnqvist and of poet Ole Torvalds, and the great-grandson of journalist and soldier Toivo Karanko.
output: 	
Torvalds | birthPlace | Helsinki, Finland
Torvalds | father | Nils Torvalds
Torvalds | mother | Anna Torvalds
Torvalds | paternalGrandfather | Leo Törnqvist
Torvalds | maternalGrandfather | Ole Torvalds
Torvalds | maternalGreatGrandfather | Toivo Karanko <DONE>

input: Torvalds attended the University of Helsinki from 1988 to 1996,[9] graduating with a master\'s degree in computer science from the NODES research group.[10] His academic career was interrupted after his first year of study when he joined the Finnish Navy Nyland Brigade in the summer of 1989, selecting the 11-month officer training program to fulfill the mandatory military service of Finland. 
output:
""",
    **parameters
)
print(f"Response from Model: {response.text}")