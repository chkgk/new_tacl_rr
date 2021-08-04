from os import environ

# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

AWS_ACCESS_KEY_ID = environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = environ.get('AWS_SECRET_ACCESS_KEY')

mturk_hit_settings = {
    'keywords': ['bonus', 'study', 'decision making'],
    'title': 'The reward is $2.5! A decision making study (~30 minutes)',
    'description': """The study takes about 30 minutes. It consists of 10 trials, played one after the other.
    All trials will be played with the same partner. You will receive $2.5 plus bonus of $1,
    based on your performance. at the end of the experiment, you will be asked to answer one simple question
    to verify you indeed followed the experiment. If you do not answer it correctly, you will not get paid.""",
    'frame_height': 700,
    'template':'global/mturk_template.html',
    #'preview_template': 'global/MTurkPreview.html',
    'minutes_allotted_per_assignment': 120,
    'expiration_hours': 2*24,  # 2 days
    # for first exp
    # 'grant_qualification_id': '3Q48MW141B1P6OKEQM2P7BMXSD2CM0',  # '3Q48MW141B1P6OKEQM2P7BMXSD2CM0',  #:mturk :sandbox, '3J48J1M4J73O984POHEYR7KF8ZK1VM',  # to prevent retakes
    # 'qualification_requirements': [
    #     {'QualificationTypeId': '3Q48MW141B1P6OKEQM2P7BMXSD2CM0',
    #      'Comparator': "DoesNotExist"},]
    # for text exp
    'grant_qualification_id': '3J48J1M4J73O984POHEYR7KF8ZK1VM',  # mturk:'3X9X2XMJQ5S9U7CCVS25R3WPXR7LQF',
    'qualification_requirements': [
        {'QualificationTypeId': '3J48J1M4J73O984POHEYR7KF8ZK1VM',
         'Comparator': "DoesNotExist"}, ]
}

SESSION_CONFIG_DEFAULTS = {
    'real_world_currency_per_point': 1,
    'participation_fee': 0.01,
    'doc': "",
    'mturk_hit_settings': mturk_hit_settings,
}

SESSION_CONFIGS = [

    {
        'name': 'text_exp_verbal_cond',
        'display_name': 'text_exp_verbal_cond',
        'num_demo_participants': 1,
        'app_sequence': ['text_exp'],
        'use_browser_bots': False,
        'cond': 'verbal',
        'review_file_name': '10_reviews',
    },
    {
        'name': 'text_exp_bot',
        'display_name': 'text_exp_bot',
        'num_demo_participants': 1,
        'app_sequence': ['text_exp_bot'],
        'use_browser_bots': False,
        'cond': 'verbal',
        'review_file_name': '10_reviews',
    },
#     {
#         'name': 'text_exp_bot_bg',
#         'display_name': 'text_exp_bot_bg',
#         'num_demo_participants': 1,
#         'app_sequence': ['text_exp_bot_bg'],
#         'use_browser_bots': False,
#         'cond': 'verbal',
#         'review_file_name': '10_reviews',
#     },
    
]

# ISO-639 code
# for example: de, fr, ja, ko, zh-hans
LANGUAGE_CODE = 'en'

# e.g. EUR, GBP, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = False

ROOMS = []


# AUTH_LEVEL:
# this setting controls which parts of your site are freely accessible,
# and which are password protected:
# - If it's not set (the default), then the whole site is freely accessible.
# - If you are launching a study and want visitors to only be able to
#   play your app if you provided them with a start link, set it to STUDY.
# - If you would like to put your site online in public demo mode where
#   anybody can play a demo version of your game, but not access the rest
#   of the admin interface, set it to DEMO.

# for flexibility, you can set it in the environment variable OTREE_AUTH_LEVEL
AUTH_LEVEL = environ.get('OTREE_AUTH_LEVEL')

ADMIN_USERNAME = 'admin'
# for security, best to set admin password in an environment variable
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')


# Consider '', None, and '0' to be empty/false
DEBUG = (environ.get('OTREE_PRODUCTION') in {None, '', '0'})

DEMO_PAGE_INTRO_HTML = """ """

# don't share this with anybody.
SECRET_KEY = 'po+8u@u7sxo_ib@7*y8ea%9%@wd6c$ojo=c@fc%v&9((9yx8wf'

# if an app is included in SESSION_CONFIGS, you don't need to list it here
INSTALLED_APPS = [
    'otree',
    'otree_mturk_utils',
    'radiogrid',
]

EXTENSION_APPS = [
    'otree_mturk_utils',
]

