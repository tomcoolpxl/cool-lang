# Coolscript Language Specification

This directory contains the official specification for the Coolscript programming language.

## Core Language
*   [Language Definition](spec/core/DEFINITION.md): The authoritative reference for syntax, semantics, and behavior.
*   [Memory Model](spec/core/DEFINITION.md#memory-model-the-no-escape-rule): Details on Move/View mechanics and the No-Escape Rule.
*   [Modules & Imports](spec/core/MODULES.md): How code is organized and shared.

## Grammar & Style
*   [PEG Grammar](spec/grammar/PEG.md): The formal grammar specification (Parsing Expression Grammar).
*   [Style Guide](spec/grammar/STYLE.md): Official formatting rules enforced by `cool fmt`.

## Standard Library
*   [Overview](spec/stdlib/OVERVIEW.md): Structure and philosophy of the Standard Library.
*   [Strings](spec/stdlib/STRINGS.md): String handling and manipulation.
*   [Iterators](spec/features/ITERATORS.md): The Iterator protocol and looping mechanics.

## Language Features
*   [Generics](../internals/compiler/GENERICS_IMPL.md): (Link to implementation for now, needs spec)
*   [Iterators](spec/features/ITERATORS.md): How `for` loops and the `Iterator` protocol work.

## Toolchain
*   [CLI Tools](spec/TOOLCHAIN.md): Documentation for the `cool` command-line interface.
