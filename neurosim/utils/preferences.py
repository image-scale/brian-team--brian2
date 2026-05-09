"""
Preference system for configuring simulator behavior.
"""

import re
from collections.abc import MutableMapping

__all__ = ['PreferenceError', 'Preference', 'PreferenceCategory', 'prefs']


class PreferenceError(Exception):
    """Exception for preference-related errors."""
    pass


class Preference:
    """
    Defines a single preference setting.

    Parameters
    ----------
    default : object
        The default value for this preference
    docs : str
        Documentation string describing the preference
    validator : callable, optional
        Function that returns True if a value is valid
    """

    def __init__(self, default, docs, validator=None):
        self.default = default
        self.docs = docs
        self._value = default
        if validator is None:
            validator = self._default_validator
        self.validator = validator

    def _default_validator(self, value):
        """Default validator checks type matches default."""
        if self.default is None:
            return True
        return isinstance(value, type(self.default))

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if not self.validator(new_value):
            raise PreferenceError(
                f"Invalid value '{new_value}' for preference "
                f"(expected type {type(self.default).__name__})"
            )
        self._value = new_value

    def reset(self):
        """Reset to default value."""
        self._value = self.default


class PreferenceCategory:
    """
    A category of related preferences.

    Allows attribute-style access to preferences within the category.
    """

    def __init__(self, name, description, parent=None):
        self._name = name
        self._description = description
        self._parent = parent
        self._preferences = {}
        self._subcategories = {}

    @property
    def name(self):
        return self._name

    @property
    def full_name(self):
        if self._parent:
            return f"{self._parent.full_name}.{self._name}"
        return self._name

    def register(self, name, preference):
        """Register a preference in this category."""
        self._validate_name(name)
        self._preferences[name] = preference

    def _validate_name(self, name):
        """Validate a preference name."""
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
            raise PreferenceError(
                f"Invalid preference name '{name}': must start with a letter "
                "and contain only letters, digits, and underscores"
            )

    def add_subcategory(self, name, description=''):
        """Add a subcategory."""
        self._validate_name(name)
        subcat = PreferenceCategory(name, description, parent=self)
        self._subcategories[name] = subcat
        return subcat

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._preferences:
            return self._preferences[name].value
        if name in self._subcategories:
            return self._subcategories[name]
        raise AttributeError(f"No preference or category '{name}' in {self._name}")

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        if name in self._preferences:
            self._preferences[name].value = value
        elif name in self._subcategories:
            raise PreferenceError(f"Cannot assign to subcategory '{name}'")
        else:
            object.__setattr__(self, name, value)

    def __contains__(self, name):
        return name in self._preferences or name in self._subcategories

    def items(self):
        """Iterate over preference name-value pairs."""
        for name, pref in self._preferences.items():
            yield name, pref.value

    def reset(self):
        """Reset all preferences to defaults."""
        for pref in self._preferences.values():
            pref.reset()
        for subcat in self._subcategories.values():
            subcat.reset()


class GlobalPreferences(MutableMapping):
    """
    Global preferences manager.

    Supports both dict-style and attribute-style access:
        prefs['core.network.schedule'] = [...]
        prefs.core.network.schedule = [...]
    """

    def __init__(self):
        self._categories = {}
        self._backup = {}

    def register_preferences(self, category_name, description, **preferences):
        """
        Register a new category of preferences.

        Parameters
        ----------
        category_name : str
            The category name (can be nested like 'core.network')
        description : str
            Description of the category
        **preferences : Preference
            Preference objects to register in this category
        """
        parts = category_name.split('.')
        current = None

        for i, part in enumerate(parts):
            if i == 0:
                if part not in self._categories:
                    self._categories[part] = PreferenceCategory(part, description)
                current = self._categories[part]
            else:
                if part not in current._subcategories:
                    current.add_subcategory(part, description)
                current = current._subcategories[part]

        for name, pref in preferences.items():
            current.register(name, pref)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._categories:
            return self._categories[name]
        raise AttributeError(f"No preference category '{name}'")

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        if name in self._categories:
            raise PreferenceError(f"Cannot assign to category '{name}'")
        object.__setattr__(self, name, value)

    def __getitem__(self, key):
        parts = key.split('.')
        current = self._categories.get(parts[0])
        if current is None:
            raise KeyError(key)

        for part in parts[1:-1]:
            current = current._subcategories.get(part)
            if current is None:
                raise KeyError(key)

        pref_name = parts[-1]
        if pref_name in current._preferences:
            return current._preferences[pref_name].value
        if pref_name in current._subcategories:
            return current._subcategories[pref_name]
        raise KeyError(key)

    def __setitem__(self, key, value):
        parts = key.split('.')
        if len(parts) < 2:
            raise PreferenceError(f"Invalid preference key: {key}")

        current = self._categories.get(parts[0])
        if current is None:
            raise KeyError(f"Unknown category: {parts[0]}")

        for part in parts[1:-1]:
            current = current._subcategories.get(part)
            if current is None:
                raise KeyError(f"Unknown subcategory: {part}")

        pref_name = parts[-1]
        if pref_name not in current._preferences:
            raise KeyError(f"Unknown preference: {pref_name}")

        current._preferences[pref_name].value = value

    def __delitem__(self, key):
        raise PreferenceError("Cannot delete preferences")

    def __iter__(self):
        for cat_name, category in self._categories.items():
            for pref_name, _ in category.items():
                yield f"{cat_name}.{pref_name}"

    def __len__(self):
        count = 0
        for category in self._categories.values():
            count += len(list(category.items()))
        return count

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def backup(self):
        """Save current preferences for later restoration."""
        self._backup = {}
        for key in self:
            self._backup[key] = self[key]

    def restore(self):
        """Restore preferences from backup."""
        for key, value in self._backup.items():
            try:
                self[key] = value
            except (KeyError, PreferenceError):
                pass

    def reset(self):
        """Reset all preferences to defaults."""
        for category in self._categories.values():
            category.reset()


# Global preferences instance
prefs = GlobalPreferences()
