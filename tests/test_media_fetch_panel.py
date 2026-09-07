import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import gradio as gr

import webui_media_fetch_panel as media_fetch_panel
from webui_panel_context import PanelContext, ReferenceTargets, CharacterTargets


class MediaFetchPanelBuildTest(unittest.TestCase):
    """Test that the media-fetch panel builds and returns expected components."""

    def test_build_media_fetch_panel_builds_and_returns_components(self):
        """build_media_fetch_panel should build inside a Blocks and return components."""
        with gr.Blocks() as demo:
            # Build the PanelContext with simple Gradio stand-ins for external refs
            ctx = PanelContext(
                reference=ReferenceTargets(
                    audio=gr.Audio(),
                    status=gr.Textbox(),
                ),
                character=CharacterTargets(
                    mode=gr.Dropdown(),
                    select=gr.Dropdown(),
                    name=gr.Textbox(),
                    summary=gr.Markdown(),
                ),
                focus_generation_tab_js="",
            )

            # Build the panel inside the Blocks context
            components = media_fetch_panel.build_media_fetch_panel(ctx)

        # Verify the returned dict has the expected keys (main controls at minimum)
        self.assertIsInstance(components, dict)
        self.assertIn("mf_url", components)
        self.assertIn("mf_run_btn", components)
        self.assertIn("mf_send_btn", components)
        self.assertIn("mf_result_path", components)
        self.assertIn("mf_status", components)
        # Additional components
        self.assertIn("mf_quality", components)
        self.assertIn("mf_format", components)
        self.assertIn("mf_progress", components)
        self.assertIn("mf_log", components)

    def test_returned_dict_is_non_empty_and_all_values_are_gradio_components(self):
        """All values in the returned dict should be gradio components or lists of components."""
        with gr.Blocks() as demo:
            ctx = PanelContext(
                reference=ReferenceTargets(
                    audio=gr.Audio(),
                    status=gr.Textbox(),
                ),
                character=CharacterTargets(
                    mode=gr.Dropdown(),
                    select=gr.Dropdown(),
                    name=gr.Textbox(),
                    summary=gr.Markdown(),
                ),
                focus_generation_tab_js="",
            )

            components = media_fetch_panel.build_media_fetch_panel(ctx)

        # Dict must be non-empty
        self.assertGreater(len(components), 0, "Returned dict should not be empty")

        # All values should be gradio components (have elem_id or be Component instances)
        # or lists/tuples of components
        for key, value in components.items():
            if isinstance(value, (list, tuple)):
                # For lists, all items should be components
                for item in value:
                    self.assertTrue(
                        hasattr(item, "elem_id") or isinstance(item, gr.components.Component),
                        f"List item in {key} is not a valid gradio component",
                    )
            else:
                self.assertTrue(
                    hasattr(value, "elem_id") or isinstance(value, gr.components.Component),
                    f"Component {key} is not a valid gradio component",
                )

    def test_dropdown_and_radio_values_are_in_choices(self):
        """For all Dropdown and Radio components, the initial value must be in choices."""
        with gr.Blocks() as demo:
            ctx = PanelContext(
                reference=ReferenceTargets(
                    audio=gr.Audio(),
                    status=gr.Textbox(),
                ),
                character=CharacterTargets(
                    mode=gr.Dropdown(),
                    select=gr.Dropdown(),
                    name=gr.Textbox(),
                    summary=gr.Markdown(),
                ),
                focus_generation_tab_js="",
            )

            components = media_fetch_panel.build_media_fetch_panel(ctx)

        # Check all Dropdown and Radio components
        for key, component in components.items():
            if isinstance(component, (gr.Dropdown, gr.Radio)):
                # Get the choices as a list of tuples or values
                choices = component.choices if hasattr(component, "choices") else []
                # Extract just the values (second element if tuple, otherwise the item itself)
                choice_values = [
                    (c[1] if isinstance(c, (tuple, list)) else c) for c in choices
                ]
                # The component's value must be in the choices or be None (for optional selects)
                if component.value is not None:
                    self.assertIn(
                        component.value,
                        choice_values,
                        f"Component {key} has value {component.value!r} not in choices {choice_values}",
                    )

    def test_slider_values_within_range(self):
        """For all Slider components, the initial value must be within min and max."""
        with gr.Blocks() as demo:
            ctx = PanelContext(
                reference=ReferenceTargets(
                    audio=gr.Audio(),
                    status=gr.Textbox(),
                ),
                character=CharacterTargets(
                    mode=gr.Dropdown(),
                    select=gr.Dropdown(),
                    name=gr.Textbox(),
                    summary=gr.Markdown(),
                ),
                focus_generation_tab_js="",
            )

            components = media_fetch_panel.build_media_fetch_panel(ctx)

        # Check all Slider components
        for key, component in components.items():
            if isinstance(component, gr.Slider):
                self.assertGreaterEqual(
                    component.value,
                    component.minimum,
                    f"Slider {key} value {component.value} is below minimum {component.minimum}",
                )
                self.assertLessEqual(
                    component.value,
                    component.maximum,
                    f"Slider {key} value {component.value} is above maximum {component.maximum}",
                )


if __name__ == "__main__":
    unittest.main()
