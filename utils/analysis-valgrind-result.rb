#!/usr/bin/env ruby

require 'rexml/document'

doc = REXML::Document.new(open(ARGV[0]))

exit 1 if doc.root.elements['error']

puts 'OK'
