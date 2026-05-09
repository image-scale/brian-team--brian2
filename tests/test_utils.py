"""
Tests for the logging and preference systems.
"""

import pytest
import logging
from io import StringIO

from neurosim.utils import (
    get_logger, SimLogger, LOG_LEVELS,
    PreferenceError, Preference, PreferenceCategory, prefs,
)


class TestLogger:
    """Tests for the logging system."""

    def setup_method(self):
        """Reset logger state before each test."""
        SimLogger._initialized = False
        SimLogger._loggers = {}
        SimLogger._console_handler = None

    def test_get_logger_returns_logger(self):
        """get_logger returns a logging.Logger instance."""
        logger = get_logger('test_module')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_module'

    def test_logger_initialization(self):
        """Logger is initialized on first use."""
        assert not SimLogger._initialized
        get_logger('test')
        assert SimLogger._initialized

    def test_log_levels_defined(self):
        """All standard log levels are defined."""
        assert 'DEBUG' in LOG_LEVELS
        assert 'INFO' in LOG_LEVELS
        assert 'WARNING' in LOG_LEVELS
        assert 'ERROR' in LOG_LEVELS
        assert 'CRITICAL' in LOG_LEVELS
        assert 'DIAGNOSTIC' in LOG_LEVELS

    def test_set_level(self):
        """Log level can be changed."""
        SimLogger.initialize()
        SimLogger.set_level('DEBUG')
        assert SimLogger._log_level == logging.DEBUG

        SimLogger.set_level('WARNING')
        assert SimLogger._log_level == logging.WARNING

    def test_logger_methods(self):
        """Logger class methods work without error."""
        SimLogger.initialize()
        SimLogger.debug('debug message')
        SimLogger.info('info message')
        SimLogger.warning('warning message')
        SimLogger.error('error message')

    def test_same_logger_returned(self):
        """Same logger is returned for same name."""
        logger1 = get_logger('mymodule')
        logger2 = get_logger('mymodule')
        assert logger1 is logger2

    def test_different_loggers_for_different_names(self):
        """Different loggers for different names."""
        logger1 = get_logger('module1')
        logger2 = get_logger('module2')
        assert logger1 is not logger2


class TestPreference:
    """Tests for the Preference class."""

    def test_preference_default_value(self):
        """Preference stores default value."""
        pref = Preference(default=10, docs='A number')
        assert pref.value == 10
        assert pref.default == 10

    def test_preference_set_value(self):
        """Preference value can be changed."""
        pref = Preference(default=10, docs='A number')
        pref.value = 20
        assert pref.value == 20

    def test_preference_type_validation(self):
        """Preference validates type by default."""
        pref = Preference(default=10, docs='An integer')
        pref.value = 20  # OK
        with pytest.raises(PreferenceError):
            pref.value = 'not an int'

    def test_preference_custom_validator(self):
        """Preference uses custom validator if provided."""
        def positive_validator(v):
            return isinstance(v, int) and v > 0

        pref = Preference(default=10, docs='Positive int', validator=positive_validator)
        pref.value = 5  # OK

        with pytest.raises(PreferenceError):
            pref.value = -5  # Not positive

    def test_preference_reset(self):
        """Preference can be reset to default."""
        pref = Preference(default=10, docs='A number')
        pref.value = 20
        assert pref.value == 20
        pref.reset()
        assert pref.value == 10

    def test_preference_docs(self):
        """Preference stores documentation."""
        docs = 'This is a description'
        pref = Preference(default=True, docs=docs)
        assert pref.docs == docs


class TestPreferenceCategory:
    """Tests for the PreferenceCategory class."""

    def test_create_category(self):
        """Category can be created with name and description."""
        cat = PreferenceCategory('core', 'Core settings')
        assert cat.name == 'core'

    def test_register_preference(self):
        """Preferences can be registered in a category."""
        cat = PreferenceCategory('core', 'Core settings')
        pref = Preference(default=0.1, docs='Time step')
        cat.register('dt', pref)
        assert 'dt' in cat

    def test_attribute_access(self):
        """Preferences can be accessed as attributes."""
        cat = PreferenceCategory('core', 'Core settings')
        cat.register('timeout', Preference(default=30, docs='Timeout'))
        assert cat.timeout == 30

    def test_attribute_set(self):
        """Preferences can be set as attributes."""
        cat = PreferenceCategory('core', 'Core settings')
        cat.register('timeout', Preference(default=30, docs='Timeout'))
        cat.timeout = 60
        assert cat.timeout == 60

    def test_invalid_name_rejected(self):
        """Invalid preference names are rejected."""
        cat = PreferenceCategory('core', 'Core settings')
        with pytest.raises(PreferenceError):
            cat.register('123invalid', Preference(default=1, docs=''))
        with pytest.raises(PreferenceError):
            cat.register('has-dash', Preference(default=1, docs=''))

    def test_subcategory(self):
        """Subcategories can be added."""
        cat = PreferenceCategory('core', 'Core settings')
        subcat = cat.add_subcategory('network', 'Network settings')
        assert 'network' in cat
        assert subcat.name == 'network'

    def test_reset_category(self):
        """Reset resets all preferences in category."""
        cat = PreferenceCategory('core', 'Core settings')
        cat.register('a', Preference(default=1, docs=''))
        cat.register('b', Preference(default=2, docs=''))
        cat.a = 100
        cat.b = 200
        cat.reset()
        assert cat.a == 1
        assert cat.b == 2


class TestGlobalPreferences:
    """Tests for the global preferences system."""

    def setup_method(self):
        """Reset prefs before each test."""
        prefs._categories = {}
        prefs._backup = {}

    def test_register_category(self):
        """Categories can be registered."""
        prefs.register_preferences(
            'test',
            'Test category',
            setting=Preference(default=True, docs='A setting')
        )
        assert 'test' in prefs._categories

    def test_attribute_access_to_category(self):
        """Categories accessible via attribute."""
        prefs.register_preferences(
            'core',
            'Core settings',
            debug=Preference(default=False, docs='Debug mode')
        )
        assert prefs.core.debug is False

    def test_nested_category(self):
        """Nested categories work correctly."""
        prefs.register_preferences(
            'core.network',
            'Network settings',
            schedule=Preference(default=['start', 'end'], docs='Schedule')
        )
        assert prefs.core.network.schedule == ['start', 'end']

    def test_dict_style_access(self):
        """Dict-style access works."""
        prefs.register_preferences(
            'test',
            'Test',
            value=Preference(default=42, docs='Value')
        )
        assert prefs['test.value'] == 42
        prefs['test.value'] = 100
        assert prefs['test.value'] == 100

    def test_invalid_key_raises(self):
        """Invalid keys raise KeyError."""
        with pytest.raises(KeyError):
            _ = prefs['nonexistent.pref']

    def test_invalid_value_raises(self):
        """Invalid values raise PreferenceError."""
        prefs.register_preferences(
            'types',
            'Type testing',
            number=Preference(default=10, docs='An integer')
        )
        with pytest.raises(PreferenceError):
            prefs.types.number = 'not a number'

    def test_contains(self):
        """__contains__ works correctly."""
        prefs.register_preferences(
            'check',
            'Check',
            exists=Preference(default=True, docs='Exists')
        )
        assert 'check.exists' in prefs
        assert 'check.missing' not in prefs

    def test_backup_and_restore(self):
        """Backup and restore work correctly."""
        prefs.register_preferences(
            'backup_test',
            'Backup test',
            val=Preference(default=1, docs='Value')
        )
        prefs.backup_test.val = 2
        prefs.backup()

        prefs.backup_test.val = 3
        assert prefs.backup_test.val == 3

        prefs.restore()
        assert prefs.backup_test.val == 2

    def test_reset_all(self):
        """Reset resets all preferences."""
        prefs.register_preferences(
            'reset_test',
            'Reset test',
            a=Preference(default=1, docs=''),
            b=Preference(default=2, docs='')
        )
        prefs.reset_test.a = 100
        prefs.reset_test.b = 200
        prefs.reset()
        assert prefs.reset_test.a == 1
        assert prefs.reset_test.b == 2

    def test_iteration(self):
        """Iteration over prefs works."""
        prefs.register_preferences(
            'iter_test',
            'Iteration test',
            x=Preference(default=1, docs=''),
            y=Preference(default=2, docs='')
        )
        keys = list(prefs)
        assert 'iter_test.x' in keys
        assert 'iter_test.y' in keys
