import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import gradio as gr
import webui_cleanup_panel
import webui_character_handlers
import webui_segmentation_handlers
from webui_panel_context import PanelContext, ReferenceTargets, CharacterTargets


class CleanupPanelBuildTests(unittest.TestCase):
    def test_build_cleanup_panel_creates_all_components(self):
        """Test that build_cleanup_panel builds all expected components inside a Blocks."""
        with gr.Blocks() as demo:
            # Create stand-in components for the PanelContext.
            upstream_path = gr.Textbox(value="", label="Upstream path")

            # Reference targets
            reference_audio = gr.Audio(label="Reference")
            reference_status = gr.Textbox(value="", label="Reference status")
            ref_targets = ReferenceTargets(
                audio=reference_audio,
                status=reference_status,
            )

            # Character targets
            character_mode = gr.Dropdown(choices=["oneshot"], value="oneshot")
            character_select = gr.Dropdown(choices=["test"], value="test")
            character_name = gr.Textbox(value="TestVoice", label="Character name")
            character_summary = gr.Markdown("Test summary")
            char_targets = CharacterTargets(
                mode=character_mode,
                select=character_select,
                name=character_name,
                summary=character_summary,
            )

            # Build context
            ctx = PanelContext(
                reference=ref_targets,
                character=char_targets,
                focus_generation_tab_js="console.log('focused');",
            )

            # Build the cleanup panel
            result = webui_cleanup_panel.build_cleanup_panel(ctx, upstream_path)

        # Assert that the returned dict contains all expected components.
        self.assertIsInstance(result, dict)

        # Main controls that are essential
        expected_keys = [
            "cl_upload",
            "cl_run_btn",
            "cl_extract_btn",
            "cl_result_path",
            "cl_voice_save_btn",
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing component: {key}")

        # Verify these are gradio components (have expected attributes)
        self.assertTrue(hasattr(result["cl_upload"], "label"))
        self.assertTrue(hasattr(result["cl_run_btn"], "label"))
        self.assertTrue(hasattr(result["cl_extract_btn"], "label"))

    def test_returned_dict_is_non_empty_and_all_values_are_gradio_components(self):
        """All values in the returned dict should be gradio components or lists of components."""
        with gr.Blocks() as demo:
            upstream_path = gr.Textbox(value="", label="Upstream path")
            ref_targets = ReferenceTargets(
                audio=gr.Audio(),
                status=gr.Textbox(),
            )
            char_targets = CharacterTargets(
                mode=gr.Dropdown(),
                select=gr.Dropdown(),
                name=gr.Textbox(),
                summary=gr.Markdown(),
            )
            ctx = PanelContext(
                reference=ref_targets,
                character=char_targets,
                focus_generation_tab_js="",
            )

            result = webui_cleanup_panel.build_cleanup_panel(ctx, upstream_path)

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
        with gr.Blocks() as demo:
            upstream_path = gr.Textbox(value="", label="Upstream path")
            ref_targets = ReferenceTargets(
                audio=gr.Audio(),
                status=gr.Textbox(),
            )
            char_targets = CharacterTargets(
                mode=gr.Dropdown(),
                select=gr.Dropdown(),
                name=gr.Textbox(),
                summary=gr.Markdown(),
            )
            ctx = PanelContext(
                reference=ref_targets,
                character=char_targets,
                focus_generation_tab_js="",
            )

            result = webui_cleanup_panel.build_cleanup_panel(ctx, upstream_path)

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
        with gr.Blocks() as demo:
            upstream_path = gr.Textbox(value="", label="Upstream path")
            ref_targets = ReferenceTargets(
                audio=gr.Audio(),
                status=gr.Textbox(),
            )
            char_targets = CharacterTargets(
                mode=gr.Dropdown(),
                select=gr.Dropdown(),
                name=gr.Textbox(),
                summary=gr.Markdown(),
            )
            ctx = PanelContext(
                reference=ref_targets,
                character=char_targets,
                focus_generation_tab_js="",
            )

            result = webui_cleanup_panel.build_cleanup_panel(ctx, upstream_path)

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

    def test_voice_save_target_caption_seeded_from_initial_character(self):
        """The save-target caption should be seeded from the initial character name."""
        with gr.Blocks() as demo:
            upstream_path = gr.Textbox(value="", label="Upstream path")
            ref_targets = ReferenceTargets(
                audio=gr.Audio(),
                status=gr.Textbox(),
            )
            char_targets = CharacterTargets(
                mode=gr.Dropdown(),
                select=gr.Dropdown(),
                name=gr.Textbox(),
                summary=gr.Markdown(),
            )
            ctx = PanelContext(
                reference=ref_targets,
                character=char_targets,
                focus_generation_tab_js="",
            )

            result = webui_cleanup_panel.build_cleanup_panel(ctx, upstream_path)

        # Get the expected caption from initial_state
        _char_mode0, _char_choices0, _char_first0, _char_name0, _char_desc0 = (
            webui_character_handlers.initial_state()
        )
        expected_caption = webui_segmentation_handlers.describe_save_target(_char_name0)

        # The cl_voice_save_target should be seeded with this caption
        self.assertIn("cl_voice_save_target", result)
        actual_caption = result["cl_voice_save_target"].value
        self.assertEqual(
            actual_caption,
            expected_caption,
            f"Save-target caption should be '{expected_caption}' but got '{actual_caption}'",
        )


if __name__ == "__main__":
    unittest.main()
