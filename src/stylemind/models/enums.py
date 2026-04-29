from __future__ import annotations

from enum import StrEnum


class BudgetTier(StrEnum):
    BUDGET = "Budget"
    MID = "Mid"
    PREMIUM = "Premium"
    LUXURY = "Luxury"


class Occasion(StrEnum):
    CASUAL = "Casual"
    OFFICE = "Office"
    DATE_NIGHT = "Date Night"
    WEEKEND_BRUNCH = "Weekend Brunch"
    ACTIVE = "Active"
    FORMAL = "Formal"


class Aesthetic(StrEnum):
    OLD_MONEY = "Old Money"
    QUIET_LUXURY = "Quiet Luxury"
    COASTAL_GRANDMA = "Coastal Grandma"
    COTTAGECORE = "Cottagecore"
    CORPORATE_MINIMALISM = "Corporate Minimalism"
    STREETWEAR = "Streetwear"
    ATHLEISURE = "Athleisure"
    Y2K = "Y2K"
    CASUAL_MINIMALISM = "Casual Minimalism"


class Season(StrEnum):
    SS = "SS"
    AW = "AW"
    YEAR_ROUND = "Year-round"


class Category(StrEnum):
    TOPS = "Tops"
    BOTTOMS = "Bottoms"
    FOOTWEAR = "Footwear"
    OUTERWEAR = "Outerwear"
    DRESSES = "Dresses"
    BAGS = "Bags"
    ACCESSORIES = "Accessories"


class BodyType(StrEnum):
    ALL = "All"
    HOURGLASS = "Hourglass"
    ATHLETIC = "Athletic"
    RECTANGLE = "Rectangle"
    PEAR = "Pear"
    INVERTED_TRIANGLE = "Inverted Triangle"


class ColorPalette(StrEnum):
    CLASSIC_NEUTRALS = "Classic Neutrals"
    EARTHY_NEUTRALS = "Earthy Neutrals"
    SOFT_WHITES = "Soft Whites"
    SPORT_MONOCHROMES = "Sport Monochromes"
    URBAN_BRIGHTS = "Urban Brights"
    WARM_NUDES = "Warm Nudes"
    MONOCHROMES = "Monochromes"
