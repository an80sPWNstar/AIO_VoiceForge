import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

import webui_segmentation_panel as panel
import webui_character_handlers
import webui_segmentation_handlers
from webui_panel_context import PanelContext, ReferenceTargets, CharacterTargets


class TestSegmentationPanel(unittest.TestCase):
    """Test the reference-clip selection panel build and wiring."""

    def test_build_segmentation_panel_creates_main_controls(self):
        """The panel builds all required controls inside a Blocks context."""
        with gr.Blocks():
            # Create stand-in components for the upstream path and context.
            upstream_path = gr.Textbox()

            # Create stand-in components for the context.
            prompt_audio = gr.Audio()
            reference_status = gr.Textbox()
            reference = ReferenceTargets(
                audio=prompt_audio,
                status=reference_status,
            )

            character_select = gr.Dropdown()
            character_name = gr.Textbox()
            character_summary = gr.Markdown()
            character = CharacterTargets(
                select=character_select,
                name=character_name,
                summary=character_summary,
            )

            focus_js = "some_js_code"
            ctx = PanelContext(
                reference=reference,
                character=character,
                focus_generation_tab_js=focus_js,
            )

            # Build the panel.
            components = panel.build_segmentation_panel(ctx, upstream_path)

            # Assert the returned dict contains the main controls.
            self.assertIn("sg_upload", components)
            self.assertIn("sg_scan_btn", components)
            self.assertIn("sg_table", components)
            self.assertIn("sg_select", components)
            self.assertIn("sg_save_btn", components)
            self.assertIn("sg_progress", components)
            self.assertIn("sg_state", components)
            self.assertIn("sg_target_voice", components)
            self.assertIn("sg_best_first", components)
            self.assertIn("sg_min_seconds", components)
            self.assertIn("sg_top_db", components)
            self.assertIn("sg_merge_gap", components)
            self.assertIn("sg_preview_audio", components)
            self.assertIn("sg_use_btn", components)
            self.assertIn("sg_status", components)
            self.assertIn("sg_detail", components)
            self.assertIn("sg_save_name", components)

    def test_build_segmentation_panel_returns_dict(self):
        """The panel returns a dict mapping component names to components."""
        with gr.Blocks():
            upstream_path = gr.Textbox()
            reference = ReferenceTargets(
                audio=gr.Audio(),
                status=gr.Textbox(),
            )
            character = CharacterTargets(
                select=gr.Dropdown(),
                name=gr.Textbox(),
                summary=gr.Markdown(),
            )
            ctx = PanelContext(
                reference=reference,
                character=character,
                focus_generation_tab_js="js_code",
            )

            result = panel.build_segmentation_panel(ctx, upstream_path)

            # Assert it's a dict.
            self.assertIsInstance(result, dict)
            # Assert all values are components (have basic gradio attributes).
            for key, value in result.items():
                self.assertTrue(
                    hasattr(value, "_id"),
                    f"Component {key} should be a gradio component",
                )

    def test_returned_dict_is_non_empty_and_all_values_are_gradio_components(self):
        """All values in the returned dict should be gradio components or lists of components."""
        with gr.Blocks():
            upstream_path = gr.Textbox()
            reference = ReferenceTargets(
                audio=gr.Audio(),
                status=gr.Textbox(),
            )
            character = CharacterTargets(
                select=gr.Dropdown(),
                name=gr.Textbox(),
                summary=gr.Markdown(),
            )
            ctx = PanelContext(
                reference=reference,
                character=character,
                focus_generation_tab_js="",
            )

            result = panel.build_segmentation_panel(ctx, upstream_path)

        # Dict must be non-empty
        self.assertGreater(len(result), 0, "Returned dict should not be empty")

        # All values should be gradio components (have elem_id or be Component instances)
        # or lists/tuples of components
        for key, value in result.items():
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
        with gr.Blocks():
            upstream_path = gr.Textbox()
            reference = ReferenceTargets(
                audio=gr.Audio(),
                status=gr.Textbox(),
            )
            character = CharacterTargets(
                select=gr.Dropdown(),
                name=gr.Textbox(),
                summary=gr.Markdown(),
            )
            ctx = PanelContext(
                reference=reference,
                character=character,
                focus_generation_tab_js="",
            )

            result = panel.build_segmentation_panel(ctx, upstream_path)

        # Check all Dropdown and Radio components
        for key, component in result.items():
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
        with gr.Blocks():
            upstream_path = gr.Textbox()
            reference = ReferenceTargets(
                audio=gr.Audio(),
                status=gr.Textbox(),
            )
            character = CharacterTargets(
                select=gr.Dropdown(),
                name=gr.Textbox(),
                summary=gr.Markdown(),
            )
            ctx = PanelContext(
                reference=reference,
                character=character,
                focus_generation_tab_js="",
            )

            result = panel.build_segmentation_panel(ctx, upstream_path)

        # Check all Slider components
        for key, component in result.items():
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

    def test_target_voice_caption_seeded_from_initial_character(self):
        """The target voice caption should be seeded from the initial character name."""
        with gr.Blocks():
            upstream_path = gr.Textbox()
            reference = ReferenceTargets(
                audio=gr.Audio(),
                status=gr.Textbox(),
            )
            character = CharacterTargets(
                select=gr.Dropdown(),
                name=gr.Textbox(),
                summary=gr.Markdown(),
            )
            ctx = PanelContext(
                reference=reference,
                character=character,
                focus_generation_tab_js="",
            )

            result = panel.build_segmentation_panel(ctx, upstream_path)

        # Get the expected caption from initial_state
        _char_choices0, _char_first0, _char_name0, _char_desc0 = (
            webui_character_handlers.initial_state()
        )
        expected_caption = webui_segmentation_handlers.describe_save_target(_char_name0)

        # The sg_target_voice should be seeded with this caption
        self.assertIn("sg_target_voice", result)
        actual_caption = result["sg_target_voice"].value
        self.assertEqual(
            actual_caption,
            expected_caption,
            f"Target voice caption should be '{expected_caption}' but got '{actual_caption}'",
        )


if __name__ == "__main__":
    unittest.main()
