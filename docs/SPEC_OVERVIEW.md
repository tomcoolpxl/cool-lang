# Coolscript Language Specification

This directory contains the official specification for the Coolscript programming language.

## Core Language
*   [Language Definition](core/DEFINITION.md): The authoritative reference for syntax, semantics, and behavior.
*   [Memory Model](core/DEFINITION.md#memory-model-the-no-escape-rule): Details on Move/View mechanics and the No-Escape Rule.
*   [Modules & Imports](core/MODULES.md): How code is organized and shared.

## Grammar & Style
*   [PEG Grammar](grammar/PEG.md): The formal grammar specification (Parsing Expression Grammar).
*   [Style Guide](grammar/STYLE.md): Official formatting rules enforced by `cool fmt`.

## Standard Library
*   [Overview](stdlib/OVERVIEW.md): Structure and philosophy of the Standard Library.
*   [Strings](stdlib/STRINGS.md): String handling and manipulation.
*   [Iterators](features/ITERATORS.md): The Iterator protocol and looping mechanics.

## Language Features
*   [Generics](../internals/compiler/GENERICS_IMPL.md): (Link to implementation for now, needs spec)
*   [Iterators](features/ITERATORS.md): How `for` loops and the `Iterator` protocol work.

## Toolchain
*   [CLI Tools](TOOLCHAIN.md): Documentation for the `cool` command-line interface.
